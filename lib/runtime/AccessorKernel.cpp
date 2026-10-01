//===- AccessorKernel.cpp - OPS accessor-style kernels -> MLIR -----------===//
//
// Translates kernels written the way most OPS applications write them,
//
//     void k(const ACC<double> &a, ACC<double> &b, const int *flags) {
//       double t = a(0,0) + a(1,0);
//       if (flags[3] == 1) b(0,0) = t;
//     }
//
// into a pure MLIR function, using the loop's argument descriptors to give the
// parameters their meaning (see KernelArgInfo). The kernel's own parameters map
// one-to-one onto the ops_par_loop arguments; the generated function's signature
// is the one xdsl_impl/ops_to_stencil.py calls:
//
//   inputs : one value per stencil point of every READ/RW dat (arg order, stencil
//            point order), then the elements of every READ global, then one
//            memref<rank x i32> per ops_arg_idx
//   outputs: one value per WRITE/RW/INC dat (arg order): the return value when
//            there is one, else stores into a trailing memref<K x T>
//
// Locals and outputs are SSA values; structured control flow is handled by
// merging them through scf.if results. Loops with compile-time bounds are
// unrolled, so accessor offsets that depend on the loop variable stay constant.
//
// Anything outside this subset makes translation fail with a message (see
// OPS_MLIR_EXPLAIN) and the loop runs through the stock OPS fallback instead.
//
//===----------------------------------------------------------------------===//

#include "runtime/KernelIRBuilder.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"

#include <array>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace ops_mlir {

namespace {

/// What the kernel sees of OPS: only the pieces its headers mention.
const char *kPrelude = R"(
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
struct ops_block_core; typedef ops_block_core *ops_block;
struct ops_dat_core; typedef ops_dat_core *ops_dat;
struct ops_reduction_core; typedef ops_reduction_core *ops_reduction;
struct ops_stencil_core; typedef ops_stencil_core *ops_stencil;
struct ops_halo_core; typedef ops_halo_core *ops_halo;
struct ops_halo_group_core; typedef ops_halo_group_core *ops_halo_group;
struct ops_arg;
typedef int ops_access;
template <class T> struct ACC {
  T &operator()(int) const; T &operator()(int, int) const; T &operator()(int, int, int) const;
};
// The same definitions as OPS's ops_macros.h -- SIGN in particular is not Fortran's.
#define MIN(a, b) ((a < b) ? (a) : (b))
#define MIN3(a, b, c) MIN(a,MIN(b,c))
#define MAX(a, b) ((a > b) ? (a) : (b))
#define MAX3(a, b, c) MAX(a,MAX(b,c))
#define SIGN(a, b) ((b < 0.0) ? (a * (-1)) : (a))
#line 1
)";

/// Something a statement can assign to.
struct Target {
  enum Kind { None, Local, Output } kind = None;
  const clang::VarDecl *var = nullptr; // Local
  int arg = -1;                        // Output
  bool operator<(const Target &o) const {
    return std::tie(kind, var, arg) < std::tie(o.kind, o.var, o.arg);
  }
};

class Translator;

/// Collects every assignable thing a statement subtree may assign to.
class AssignScan : public clang::RecursiveASTVisitor<AssignScan> {
public:
  explicit AssignScan(Translator &t) : tr(t) {}
  bool VisitBinaryOperator(clang::BinaryOperator *op);
  bool VisitUnaryOperator(clang::UnaryOperator *op);
  std::set<Target> targets;

private:
  Translator &tr;
};

class Translator {
public:
  Translator(mlir::MLIRContext &ctx, const clang::FunctionDecl *decl,
             const std::vector<KernelArgInfo> &args, int rank,
             const std::map<std::string, const void *> &constants,
             llvm::raw_ostream &errs)
      : ctx(ctx), b(&ctx), loc(b.getUnknownLoc()), decl(decl), args(args),
        rank(rank), constants(constants), errs(errs), astCtx(decl->getASTContext()) {}

  mlir::func::FuncOp run(const std::string &name);
  const std::vector<KernelConstRef> &constRefs() const { return constRefList; }

  Target resolveTarget(const clang::Expr *e);
  bool ok() const { return ok_; }

private:
  mlir::MLIRContext &ctx;
  mlir::OpBuilder b;
  mlir::Location loc;
  const clang::FunctionDecl *decl;
  const std::vector<KernelArgInfo> &args;
  int rank;
  const std::map<std::string, const void *> &constants;
  llvm::raw_ostream &errs;
  clang::ASTContext &astCtx;
  bool ok_ = true;

  // value of every (argument, stencil point) and of every global element
  std::vector<std::vector<mlir::Value>> pointValue;
  std::vector<std::vector<mlir::Value>> gblValue;
  std::vector<mlir::Value> idxMemref;
  llvm::DenseMap<const clang::ParmVarDecl *, int> parmIndex;

  struct Env {
    llvm::DenseMap<const clang::VarDecl *, mlir::Value> locals;
    std::vector<mlir::Value> out; // current value of each output argument (or null)
  } env;
  llvm::DenseMap<const clang::VarDecl *, int64_t> constVars; // unrolled loop variables

  // Registered constants the body reads, in order of first use, with the function
  // parameter that carries each.
  struct ConstUse {
    const clang::VarDecl *var;
    int64_t offset;
    mlir::Type elt;
    mlir::Value value;
  };
  std::vector<ConstUse> constUses;
  std::vector<KernelConstRef> constRefList;
  void collectConstUses();
  /// The (global, byte offset) a `g` or `g.member.member` expression names.
  bool constantRef(const clang::Expr *e, const clang::VarDecl *&var, int64_t &offset,
                   clang::QualType &type);
  mlir::Value constantValue(const clang::Expr *e);

  //---- helpers -------------------------------------------------------------
  mlir::Value fail(const llvm::Twine &msg) {
    if (ok_) {
      ok_ = false;
      errs << "accessor kernel '" << decl->getNameAsString() << "': " << msg << "\n";
    }
    return {};
  }

  mlir::Type scalarType(clang::QualType t) {
    t = t.getNonReferenceType().getUnqualifiedType();
    if (t->isSpecificBuiltinType(clang::BuiltinType::Double))
      return b.getF64Type();
    if (t->isSpecificBuiltinType(clang::BuiltinType::Float))
      return b.getF32Type();
    if (t->isBooleanType())
      return b.getI1Type();
    if (t->isIntegerType())
      return b.getIntegerType(astCtx.getTypeSize(t));
    return {};
  }

  mlir::Value constFloat(mlir::Type t, double v) {
    return b.create<mlir::arith::ConstantOp>(loc, b.getFloatAttr(t, v));
  }
  mlir::Value constInt(mlir::Type t, int64_t v) {
    return b.create<mlir::arith::ConstantOp>(loc, b.getIntegerAttr(t, v));
  }
  mlir::Value zeroOf(mlir::Type t) {
    return mlir::isa<mlir::FloatType>(t) ? constFloat(t, 0.0) : constInt(t, 0);
  }

  mlir::Value cast(mlir::Value v, mlir::Type to) {
    if (!v || v.getType() == to)
      return v;
    mlir::Type from = v.getType();
    bool fromF = mlir::isa<mlir::FloatType>(from), toF = mlir::isa<mlir::FloatType>(to);
    if (fromF && toF)
      return from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth()
                 ? (mlir::Value)b.create<mlir::arith::ExtFOp>(loc, to, v)
                 : (mlir::Value)b.create<mlir::arith::TruncFOp>(loc, to, v);
    if (!fromF && toF)
      return b.create<mlir::arith::SIToFPOp>(loc, to, v);
    if (fromF && !toF)
      return b.create<mlir::arith::FPToSIOp>(loc, to, v);
    if (to.isInteger(1)) // int -> bool
      return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, v,
                                           constInt(from, 0));
    if (from.isInteger(1))
      return b.create<mlir::arith::ExtUIOp>(loc, to, v);
    return from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth()
               ? (mlir::Value)b.create<mlir::arith::ExtSIOp>(loc, to, v)
               : (mlir::Value)b.create<mlir::arith::TruncIOp>(loc, to, v);
  }

  mlir::Value toBool(mlir::Value v) { return cast(v, b.getI1Type()); }

  /// Compile-time integer value: literals, enumerators, unrolled loop variables and
  /// registered extern constants.
  std::optional<int64_t> evalInt(const clang::Expr *e) {
    e = e->IgnoreParenImpCasts();
    if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
      if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl())) {
        auto it = constVars.find(vd);
        if (it != constVars.end())
          return it->second;
        if (vd->hasGlobalStorage()) {
          auto c = constants.find(vd->getNameAsString());
          if (c != constants.end() && c->second && vd->getType()->isIntegerType())
            return *reinterpret_cast<const int32_t *>(c->second);
        }
      }
    }
    if (const auto *un = llvm::dyn_cast<clang::UnaryOperator>(e)) {
      auto v = evalInt(un->getSubExpr());
      if (v && un->getOpcode() == clang::UO_Minus)
        return -*v;
      if (v && un->getOpcode() == clang::UO_Plus)
        return *v;
    }
    if (const auto *bin = llvm::dyn_cast<clang::BinaryOperator>(e)) {
      auto l = evalInt(bin->getLHS()), r = evalInt(bin->getRHS());
      if (l && r) {
        switch (bin->getOpcode()) {
        case clang::BO_Add: return *l + *r;
        case clang::BO_Sub: return *l - *r;
        case clang::BO_Mul: return *l * *r;
        default: break;
        }
      }
    }
    clang::Expr::EvalResult res;
    if (e->EvaluateAsInt(res, astCtx))
      return res.Val.getInt().getSExtValue();
    return std::nullopt;
  }

  /// The argument a (possibly parenthesised / cast) reference to a parameter names.
  int paramArg(const clang::Expr *e) {
    e = e->IgnoreParenImpCasts();
    if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e))
      if (const auto *pv = llvm::dyn_cast<clang::ParmVarDecl>(dre->getDecl())) {
        auto it = parmIndex.find(pv);
        if (it != parmIndex.end())
          return it->second;
      }
    return -1;
  }

  //---- expressions ---------------------------------------------------------
  mlir::Value readAccessor(const clang::CXXOperatorCallExpr *call);
  mlir::Value emitCall(const clang::CallExpr *call);
  mlir::Value emitBinary(const clang::BinaryOperator *op);
  mlir::Value emit(const clang::Expr *e);

  //---- statements ----------------------------------------------------------
  void assign(const Target &t, mlir::Value v);
  mlir::Value current(const Target &t);
  void emitStmt(const clang::Stmt *s);
  void emitIf(const clang::IfStmt *s);
  void emitFor(const clang::ForStmt *s);
  void emitAssignment(const clang::BinaryOperator *op);
  void emitIncDec(const clang::UnaryOperator *op);

  friend class AssignScan;
};

//===----------------------------------------------------------------------===//
// Assignment scanning
//===----------------------------------------------------------------------===//

bool AssignScan::VisitBinaryOperator(clang::BinaryOperator *op) {
  if (op->isAssignmentOp()) {
    Target t = tr.resolveTarget(op->getLHS());
    if (t.kind != Target::None)
      targets.insert(t);
  }
  return true;
}

bool AssignScan::VisitUnaryOperator(clang::UnaryOperator *op) {
  if (op->isIncrementDecrementOp()) {
    Target t = tr.resolveTarget(op->getSubExpr());
    if (t.kind != Target::None)
      targets.insert(t);
  }
  return true;
}

Target Translator::resolveTarget(const clang::Expr *e) {
  e = e->IgnoreParenImpCasts();
  Target t;
  if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
    if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl());
        vd && !llvm::isa<clang::ParmVarDecl>(vd) && !vd->hasGlobalStorage()) {
      t.kind = Target::Local;
      t.var = vd;
    }
  } else if (const auto *call = llvm::dyn_cast<clang::CXXOperatorCallExpr>(e)) {
    if (call->getOperator() == clang::OO_Call && call->getNumArgs() >= 1) {
      int a = paramArg(call->getArg(0));
      if (a >= 0) {
        t.kind = Target::Output;
        t.arg = a;
      }
    }
  }
  return t;
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

bool Translator::constantRef(const clang::Expr *e, const clang::VarDecl *&var,
                             int64_t &offset, clang::QualType &type) {
  e = e->IgnoreParenImpCasts();
  if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
    const auto *vd = llvm::dyn_cast<clang::VarDecl>(dre->getDecl());
    if (!vd || llvm::isa<clang::ParmVarDecl>(vd) || !vd->hasGlobalStorage())
      return false;
    var = vd;
    offset = 0;
    type = vd->getType();
    return true;
  }
  if (const auto *me = llvm::dyn_cast<clang::MemberExpr>(e)) {
    if (me->isArrow())
      return false;
    const auto *fd = llvm::dyn_cast<clang::FieldDecl>(me->getMemberDecl());
    if (!fd || !constantRef(me->getBase(), var, offset, type))
      return false;
    const clang::RecordDecl *rec = fd->getParent();
    if (rec->isInvalidDecl())
      return false;
    offset += astCtx.getASTRecordLayout(rec).getFieldOffset(fd->getFieldIndex()) / 8;
    type = fd->getType();
    return true;
  }
  return false;
}

void Translator::collectConstUses() {
  struct Collector : clang::RecursiveASTVisitor<Collector> {
    Translator &tr;
    explicit Collector(Translator &t) : tr(t) {}
    void note(clang::Expr *e) {
      const clang::VarDecl *var = nullptr;
      int64_t offset = 0;
      clang::QualType type;
      if (!tr.constantRef(e, var, offset, type))
        return;
      mlir::Type elt = tr.scalarType(type);
      if (!elt || elt.isInteger(1) || !tr.constants.count(var->getNameAsString()))
        return;
      for (const ConstUse &u : tr.constUses)
        if (u.var == var && u.offset == offset)
          return;
      tr.constUses.push_back({var, offset, elt, {}});
    }
    bool VisitDeclRefExpr(clang::DeclRefExpr *e) { return note(e), true; }
    bool VisitMemberExpr(clang::MemberExpr *e) { return note(e), true; }
  } collector(*this);
  collector.TraverseStmt(const_cast<clang::Stmt *>(decl->getBody()));
}

mlir::Value Translator::constantValue(const clang::Expr *e) {
  const clang::VarDecl *var = nullptr;
  int64_t offset = 0;
  clang::QualType type;
  if (!constantRef(e, var, offset, type))
    return fail("unsupported reference to a global or a member of one");
  auto reg = constants.find(var->getNameAsString());
  if (reg == constants.end() || !reg->second)
    return fail("reference to '" + var->getNameAsString() +
                "': not a registered constant (ops_register_kernel_constant)");
  for (const ConstUse &u : constUses)
    if (u.var == var && u.offset == offset)
      return u.value;
  return fail("global '" + var->getNameAsString() + "' has an unsupported type");
}

mlir::Value Translator::readAccessor(const clang::CXXOperatorCallExpr *call) {
  int a = paramArg(call->getArg(0));
  if (a < 0)
    return fail("accessor call on something that is not a kernel argument");
  const KernelArgInfo &info = args[a];
  std::array<int, 3> off = {0, 0, 0};
  unsigned n = call->getNumArgs() - 1;
  if (n != static_cast<unsigned>(rank))
    return fail("accessor with " + llvm::Twine(n) + " indices in a " + llvm::Twine(rank) +
                "D loop (multi-component dats are not supported)");
  for (unsigned i = 0; i < n; ++i) {
    auto v = evalInt(call->getArg(i + 1));
    if (!v)
      return fail("accessor offset is not a compile-time constant");
    off[i] = static_cast<int>(*v);
  }
  bool zero = off[0] == 0 && off[1] == 0 && off[2] == 0;
  if (zero && info.access != 0 /*OPS_READ*/ && env.out[a])
    return env.out[a]; // reads back a value this kernel already wrote
  for (size_t p = 0; p < info.points.size(); ++p)
    if (info.points[p] == off && !pointValue[a].empty())
      return pointValue[a][p];
  return fail("argument " + llvm::Twine(a) + " is read at offset (" + llvm::Twine(off[0]) + "," +
              llvm::Twine(off[1]) + "," + llvm::Twine(off[2]) + ") which is not a point of its stencil");
}

mlir::Value Translator::emitCall(const clang::CallExpr *call) {
  if (const auto *op = llvm::dyn_cast<clang::CXXOperatorCallExpr>(call)) {
    if (op->getOperator() == clang::OO_Call)
      return readAccessor(op);
    return fail("unsupported overloaded operator");
  }
  const clang::FunctionDecl *callee = call->getDirectCallee();
  if (!callee || !callee->getDeclName().isIdentifier())
    return fail("call to an unknown function");
  llvm::StringRef name = callee->getName();
  if (name.ends_with("f") && name != "fabs")
    name = name.drop_back(); // sinf -> sin
  llvm::SmallVector<mlir::Value, 2> a;
  for (const clang::Expr *arg : call->arguments()) {
    mlir::Value v = emit(arg);
    if (!v)
      return {};
    a.push_back(v);
  }
  auto un = [&](auto tag) -> mlir::Value {
    using Op = decltype(tag);
    return b.create<Op>(loc, a[0]);
  };
  if (a.size() == 1) {
    if (name == "sqrt") return un(mlir::math::SqrtOp());
    if (name == "fabs") return un(mlir::math::AbsFOp());
    if (name == "exp") return un(mlir::math::ExpOp());
    if (name == "log") return un(mlir::math::LogOp());
    if (name == "sin") return un(mlir::math::SinOp());
    if (name == "cos") return un(mlir::math::CosOp());
    if (name == "tan") return un(mlir::math::TanOp());
    if (name == "floor") return un(mlir::math::FloorOp());
    if (name == "ceil") return un(mlir::math::CeilOp());
    if (name == "atan") return un(mlir::math::AtanOp());
  } else if (a.size() == 2) {
    if (name == "pow") return b.create<mlir::math::PowFOp>(loc, a[0], a[1]);
    if (name == "fmin") return b.create<mlir::arith::MinNumFOp>(loc, a[0], a[1]);
    if (name == "fmax") return b.create<mlir::arith::MaxNumFOp>(loc, a[0], a[1]);
  }
  return fail("call to unsupported function '" + callee->getNameAsString() + "'");
}

mlir::Value Translator::emitBinary(const clang::BinaryOperator *op) {
  using P = mlir::arith::CmpFPredicate;
  using I = mlir::arith::CmpIPredicate;
  if (op->getOpcode() == clang::BO_LAnd || op->getOpcode() == clang::BO_LOr) {
    mlir::Value l = toBool(emit(op->getLHS())), r = toBool(emit(op->getRHS()));
    if (!l || !r)
      return {};
    return op->getOpcode() == clang::BO_LAnd
               ? (mlir::Value)b.create<mlir::arith::AndIOp>(loc, l, r)
               : (mlir::Value)b.create<mlir::arith::OrIOp>(loc, l, r);
  }
  mlir::Value l = emit(op->getLHS()), r = emit(op->getRHS());
  if (!l || !r)
    return {};
  if (l.getType() != r.getType()) // e.g. an i1 compared with an int
    r = cast(r, l.getType());
  bool f = mlir::isa<mlir::FloatType>(l.getType());
  auto cmp = [&](P fp, I ip) -> mlir::Value {
    return f ? (mlir::Value)b.create<mlir::arith::CmpFOp>(loc, fp, l, r)
             : (mlir::Value)b.create<mlir::arith::CmpIOp>(loc, ip, l, r);
  };
  switch (op->getOpcode()) {
  case clang::BO_Add:
    return f ? (mlir::Value)b.create<mlir::arith::AddFOp>(loc, l, r)
             : (mlir::Value)b.create<mlir::arith::AddIOp>(loc, l, r);
  case clang::BO_Sub:
    return f ? (mlir::Value)b.create<mlir::arith::SubFOp>(loc, l, r)
             : (mlir::Value)b.create<mlir::arith::SubIOp>(loc, l, r);
  case clang::BO_Mul:
    return f ? (mlir::Value)b.create<mlir::arith::MulFOp>(loc, l, r)
             : (mlir::Value)b.create<mlir::arith::MulIOp>(loc, l, r);
  case clang::BO_Div:
    return f ? (mlir::Value)b.create<mlir::arith::DivFOp>(loc, l, r)
             : (mlir::Value)b.create<mlir::arith::DivSIOp>(loc, l, r);
  case clang::BO_Rem:
    return f ? fail("floating-point %") : (mlir::Value)b.create<mlir::arith::RemSIOp>(loc, l, r);
  case clang::BO_LT: return cmp(P::OLT, I::slt);
  case clang::BO_LE: return cmp(P::OLE, I::sle);
  case clang::BO_GT: return cmp(P::OGT, I::sgt);
  case clang::BO_GE: return cmp(P::OGE, I::sge);
  case clang::BO_EQ: return cmp(P::OEQ, I::eq);
  case clang::BO_NE: return cmp(P::UNE, I::ne);
  default: return fail("unsupported binary operator");
  }
}

mlir::Value Translator::emit(const clang::Expr *e) {
  if (!ok_ || !e)
    return {};
  if (const auto *x = llvm::dyn_cast<clang::ParenExpr>(e))
    return emit(x->getSubExpr());
  if (const auto *x = llvm::dyn_cast<clang::ImplicitCastExpr>(e)) {
    mlir::Value v = emit(x->getSubExpr());
    if (!v)
      return {};
    switch (x->getCastKind()) {
    case clang::CK_LValueToRValue: case clang::CK_NoOp: case clang::CK_IntegralToBoolean:
    case clang::CK_FloatingToBoolean: case clang::CK_ArrayToPointerDecay:
    case clang::CK_FunctionToPointerDecay: case clang::CK_BuiltinFnToFnPtr:
      if (x->getCastKind() == clang::CK_IntegralToBoolean || x->getCastKind() == clang::CK_FloatingToBoolean)
        return cast(v, b.getI1Type());
      return v;
    default: {
      mlir::Type to = scalarType(x->getType());
      if (!to)
        return fail("unsupported implicit conversion");
      return cast(v, to);
    }
    }
  }
  if (llvm::isa<clang::MemberExpr>(e))
    return constantValue(e);
  if (const auto *x = llvm::dyn_cast<clang::CStyleCastExpr>(e)) {
    mlir::Type to = scalarType(x->getType());
    mlir::Value v = emit(x->getSubExpr());
    return to ? cast(v, to) : fail("unsupported cast");
  }
  if (const auto *x = llvm::dyn_cast<clang::IntegerLiteral>(e))
    return constInt(scalarType(x->getType()), x->getValue().getSExtValue());
  if (const auto *x = llvm::dyn_cast<clang::FloatingLiteral>(e))
    return constFloat(scalarType(x->getType()), x->getValueAsApproximateDouble());
  if (const auto *x = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(e))
    return constInt(b.getI1Type(), x->getValue());
  if (const auto *x = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
    const clang::ValueDecl *d = x->getDecl();
    if (const auto *en = llvm::dyn_cast<clang::EnumConstantDecl>(d))
      return constInt(b.getI32Type(), en->getInitVal().getSExtValue());
    if (const auto *vd = llvm::dyn_cast<clang::VarDecl>(d)) {
      auto cv = constVars.find(vd);
      if (cv != constVars.end())
        return constInt(b.getI32Type(), cv->second);
      auto it = env.locals.find(vd);
      if (it != env.locals.end())
        return it->second;
      if (vd->hasGlobalStorage())
        return constantValue(e);
      if (llvm::isa<clang::ParmVarDecl>(vd)) {
        int a = paramArg(e);
        // a by-value global scalar
        if (a >= 0 && args[a].kind == KernelArgInfo::Kind::Gbl && !gblValue[a].empty() &&
            !vd->getType()->isPointerType())
          return gblValue[a][0];
        return fail("kernel parameter '" + vd->getNameAsString() + "' used in an unsupported way");
      }
      return fail("variable '" + vd->getNameAsString() + "' read before it is defined");
    }
    return fail("unsupported declaration reference");
  }
  if (const auto *x = llvm::dyn_cast<clang::UnaryOperator>(e)) {
    if (x->getOpcode() == clang::UO_Deref) {
      int a = paramArg(x->getSubExpr());
      if (a >= 0 && args[a].kind == KernelArgInfo::Kind::Gbl && !gblValue[a].empty())
        return gblValue[a][0];
      return fail("dereference of something other than a global argument");
    }
    mlir::Value v = emit(x->getSubExpr());
    if (!v)
      return {};
    switch (x->getOpcode()) {
    case clang::UO_Minus:
      return mlir::isa<mlir::FloatType>(v.getType())
                 ? (mlir::Value)b.create<mlir::arith::NegFOp>(loc, v)
                 : (mlir::Value)b.create<mlir::arith::SubIOp>(loc, zeroOf(v.getType()), v);
    case clang::UO_Plus: return v;
    case clang::UO_LNot: {
      mlir::Value bv = toBool(v);
      return b.create<mlir::arith::XOrIOp>(loc, bv, constInt(b.getI1Type(), 1));
    }
    default: return fail("unsupported unary operator");
    }
  }
  if (const auto *x = llvm::dyn_cast<clang::BinaryOperator>(e))
    return emitBinary(x);
  if (const auto *x = llvm::dyn_cast<clang::ConditionalOperator>(e)) {
    mlir::Value c = toBool(emit(x->getCond())), t = emit(x->getTrueExpr()),
                f = emit(x->getFalseExpr());
    if (!c || !t || !f)
      return {};
    if (t.getType() != f.getType())
      f = cast(f, t.getType());
    return b.create<mlir::arith::SelectOp>(loc, c, t, f);
  }
  if (const auto *x = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
    int a = paramArg(x->getBase());
    auto idx = evalInt(x->getIdx());
    if (a >= 0 && idx && args[a].kind == KernelArgInfo::Kind::Gbl && *idx >= 0 &&
        static_cast<size_t>(*idx) < gblValue[a].size())
      return gblValue[a][*idx];
    return fail("unsupported array subscript");
  }
  if (const auto *x = llvm::dyn_cast<clang::CallExpr>(e))
    return emitCall(x);
  return fail(llvm::Twine("unsupported expression (") + e->getStmtClassName() + ")");
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

mlir::Value Translator::current(const Target &t) {
  if (t.kind == Target::Local) {
    auto it = env.locals.find(t.var);
    return it == env.locals.end() ? mlir::Value() : it->second;
  }
  return env.out[t.arg];
}

void Translator::assign(const Target &t, mlir::Value v) {
  if (!v)
    return;
  if (t.kind == Target::Local) {
    mlir::Type ty = scalarType(t.var->getType());
    env.locals[t.var] = ty ? cast(v, ty) : v;
  } else if (t.kind == Target::Output) {
    env.out[t.arg] = cast(v, args[t.arg].elt);
  }
}

void Translator::emitAssignment(const clang::BinaryOperator *op) {
  Target t = resolveTarget(op->getLHS());
  if (t.kind == Target::None) {
    fail("assignment to something other than a local or an output accessor");
    return;
  }
  if (t.kind == Target::Output) {
    // only the centre point can be written
    const auto *call = llvm::cast<clang::CXXOperatorCallExpr>(op->getLHS()->IgnoreParenImpCasts());
    for (unsigned i = 1; i < call->getNumArgs(); ++i) {
      auto v = evalInt(call->getArg(i));
      if (!v || *v != 0) {
        fail("output accessor written at a non-zero offset");
        return;
      }
    }
    if (args[t.arg].access == 0 /*OPS_READ*/) {
      fail("write to a READ argument");
      return;
    }
  }
  mlir::Value rhs = emit(op->getRHS());
  if (!rhs)
    return;
  if (op->getOpcode() != clang::BO_Assign) {
    mlir::Value cur = t.kind == Target::Output && !env.out[t.arg] ? mlir::Value() : current(t);
    if (!cur) {
      fail("compound assignment to a value with no prior value");
      return;
    }
    rhs = cast(rhs, cur.getType());
    bool f = mlir::isa<mlir::FloatType>(cur.getType());
    switch (op->getOpcode()) {
    case clang::BO_AddAssign:
      rhs = f ? (mlir::Value)b.create<mlir::arith::AddFOp>(loc, cur, rhs)
              : (mlir::Value)b.create<mlir::arith::AddIOp>(loc, cur, rhs);
      break;
    case clang::BO_SubAssign:
      rhs = f ? (mlir::Value)b.create<mlir::arith::SubFOp>(loc, cur, rhs)
              : (mlir::Value)b.create<mlir::arith::SubIOp>(loc, cur, rhs);
      break;
    case clang::BO_MulAssign:
      rhs = f ? (mlir::Value)b.create<mlir::arith::MulFOp>(loc, cur, rhs)
              : (mlir::Value)b.create<mlir::arith::MulIOp>(loc, cur, rhs);
      break;
    case clang::BO_DivAssign:
      rhs = f ? (mlir::Value)b.create<mlir::arith::DivFOp>(loc, cur, rhs)
              : (mlir::Value)b.create<mlir::arith::DivSIOp>(loc, cur, rhs);
      break;
    default:
      fail("unsupported compound assignment");
      return;
    }
  }
  assign(t, rhs);
}

void Translator::emitIncDec(const clang::UnaryOperator *op) {
  Target t = resolveTarget(op->getSubExpr());
  mlir::Value cur = t.kind == Target::Local ? current(t) : mlir::Value();
  if (!cur) {
    fail("++/-- on something other than a local");
    return;
  }
  mlir::Value one = mlir::isa<mlir::FloatType>(cur.getType()) ? constFloat(cur.getType(), 1.0)
                                                              : constInt(cur.getType(), 1);
  bool inc = op->isIncrementOp(), f = mlir::isa<mlir::FloatType>(cur.getType());
  mlir::Value r = inc ? (f ? (mlir::Value)b.create<mlir::arith::AddFOp>(loc, cur, one)
                           : (mlir::Value)b.create<mlir::arith::AddIOp>(loc, cur, one))
                      : (f ? (mlir::Value)b.create<mlir::arith::SubFOp>(loc, cur, one)
                           : (mlir::Value)b.create<mlir::arith::SubIOp>(loc, cur, one));
  assign(t, r);
}

void Translator::emitIf(const clang::IfStmt *s) {
  mlir::Value cond = toBool(emit(s->getCond()));
  if (!cond)
    return;
  AssignScan scan(*this);
  scan.TraverseStmt(const_cast<clang::Stmt *>(s->getThen()));
  if (s->getElse())
    scan.TraverseStmt(const_cast<clang::Stmt *>(s->getElse()));

  std::vector<Target> targets;
  std::vector<mlir::Value> before;
  std::vector<mlir::Type> types;
  for (const Target &t : scan.targets) {
    mlir::Value cur = current(t);
    if (!cur) {
      if (t.kind == Target::Local)
        continue; // declared inside the branch
      fail("an output written conditionally has no value to keep on the other path "
           "(argument " + llvm::Twine(t.arg) + ")");
      return;
    }
    targets.push_back(t);
    before.push_back(cur);
    types.push_back(cur.getType());
  }

  // Both branches only compute on values that are already loaded, so they can run
  // unconditionally and be merged with selects. That keeps the kernel body free of
  // regions: a region inside the memref.alloca_scope the kernel is inlined into
  // would have to become several blocks, which the scope forbids. Integer division
  // is the one operation that must not run speculatively.
  struct Unsafe : clang::RecursiveASTVisitor<Unsafe> {
    bool found = false;
    bool VisitBinaryOperator(clang::BinaryOperator *op) {
      switch (op->getOpcode()) {
      case clang::BO_Div: case clang::BO_Rem: case clang::BO_DivAssign: case clang::BO_RemAssign:
        if (op->getType()->isIntegerType())
          found = true;
        break;
      default: break;
      }
      return true;
    }
  } unsafe;
  unsafe.TraverseStmt(const_cast<clang::Stmt *>(s->getThen()));
  if (s->getElse())
    unsafe.TraverseStmt(const_cast<clang::Stmt *>(s->getElse()));
  if (unsafe.found) {
    fail("integer division inside a conditional cannot be evaluated speculatively");
    return;
  }

  Env saved = env;
  auto branch = [&](const clang::Stmt *body) {
    env = saved;
    if (body)
      emitStmt(body);
    std::vector<mlir::Value> vals;
    for (size_t i = 0; i < targets.size(); ++i)
      vals.push_back(cast(current(targets[i]), types[i]));
    return vals;
  };
  std::vector<mlir::Value> thenVals = branch(s->getThen());
  std::vector<mlir::Value> elseVals = branch(s->getElse());
  env = saved;
  if (!ok_)
    return;
  for (size_t i = 0; i < targets.size(); ++i)
    assign(targets[i], b.create<mlir::arith::SelectOp>(loc, cond, thenVals[i], elseVals[i]));
}

void Translator::emitFor(const clang::ForStmt *s) {
  // for (int v = a; v < b; v++) { ... } with compile-time a and b: unrolled.
  const auto *init = llvm::dyn_cast_or_null<clang::DeclStmt>(s->getInit());
  const auto *cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(s->getCond());
  const auto *inc = llvm::dyn_cast_or_null<clang::UnaryOperator>(s->getInc());
  if (!init || !init->isSingleDecl() || !cond || !inc || !inc->isIncrementOp()) {
    fail("loop is not of the form for (int v = a; v < b; v++)");
    return;
  }
  const auto *var = llvm::dyn_cast<clang::VarDecl>(init->getSingleDecl());
  if (!var || !var->hasInit()) {
    fail("loop variable has no initialiser");
    return;
  }
  auto lo = evalInt(var->getInit());
  auto hi = evalInt(cond->getRHS());
  if (!lo || !hi || (cond->getOpcode() != clang::BO_LT && cond->getOpcode() != clang::BO_LE)) {
    fail("loop bounds are not compile-time constants");
    return;
  }
  int64_t end = cond->getOpcode() == clang::BO_LE ? *hi + 1 : *hi;
  if (end - *lo > 256) {
    fail("loop would unroll to more than 256 iterations");
    return;
  }
  for (int64_t i = *lo; i < end && ok_; ++i) {
    constVars[var] = i;
    emitStmt(s->getBody());
  }
  constVars.erase(var);
}

void Translator::emitStmt(const clang::Stmt *s) {
  if (!ok_ || !s)
    return;
  if (const auto *c = llvm::dyn_cast<clang::CompoundStmt>(s)) {
    for (const clang::Stmt *x : c->body())
      emitStmt(x);
  } else if (const auto *d = llvm::dyn_cast<clang::DeclStmt>(s)) {
    for (const clang::Decl *decl : d->decls()) {
      const auto *vd = llvm::dyn_cast<clang::VarDecl>(decl);
      if (!vd) {
        fail("unsupported declaration");
        return;
      }
      mlir::Type ty = scalarType(vd->getType());
      if (!ty || vd->getType()->isArrayType() || vd->getType()->isPointerType()) {
        fail("local '" + vd->getNameAsString() + "' has an unsupported type");
        return;
      }
      mlir::Value v = vd->hasInit() ? cast(emit(vd->getInit()), ty) : zeroOf(ty);
      if (v)
        env.locals[vd] = v;
    }
  } else if (const auto *x = llvm::dyn_cast<clang::IfStmt>(s)) {
    emitIf(x);
  } else if (const auto *x = llvm::dyn_cast<clang::ForStmt>(s)) {
    emitFor(x);
  } else if (const auto *x = llvm::dyn_cast<clang::BinaryOperator>(s)) {
    if (x->isAssignmentOp())
      emitAssignment(x);
    else
      fail("expression statement without effect");
  } else if (const auto *x = llvm::dyn_cast<clang::UnaryOperator>(s)) {
    if (x->isIncrementDecrementOp())
      emitIncDec(x);
    else
      fail("expression statement without effect");
  } else if (const auto *x = llvm::dyn_cast<clang::ReturnStmt>(s)) {
    if (x->getRetValue())
      fail("kernel returns a value");
    // a trailing `return;` is fine; an early one is not handled (rare)
  } else if (llvm::isa<clang::NullStmt>(s)) {
  } else if (const auto *x = llvm::dyn_cast<clang::ExprWithCleanups>(s)) {
    emitStmt(x->getSubExpr());
  } else {
    fail(llvm::Twine("unsupported statement (") + s->getStmtClassName() + ")");
  }
}

//===----------------------------------------------------------------------===//
// Function assembly
//===----------------------------------------------------------------------===//

mlir::func::FuncOp Translator::run(const std::string &name) {
  const auto params = decl->parameters();
  if (params.size() != args.size()) {
    fail("kernel has " + llvm::Twine(params.size()) + " parameters but the loop passes " +
         llvm::Twine(args.size()) + " arguments");
    return {};
  }
  for (size_t i = 0; i < params.size(); ++i)
    parmIndex[params[i]] = static_cast<int>(i);

  collectConstUses();

  // signature: dat points, then global elements, then constants, then the outputs.
  // The loop's other lowering (ops_to_stencil.py) calls kernels in this order.
  std::vector<mlir::Type> inputs, gblInputs;
  std::vector<std::pair<int, size_t>> pointSlots; // (arg, point) per input
  std::vector<std::pair<int, size_t>> gblSlots;
  std::vector<int> idxSlots;
  mlir::Type outType;
  int numOutputs = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    const KernelArgInfo &k = args[i];
    switch (k.kind) {
    case KernelArgInfo::Kind::Dat:
      if (!k.elt)
        return fail("dat of unsupported element type"), mlir::func::FuncOp();
      if (k.strided)
        return fail("strided stencils (dats indexed along fewer dimensions) are not supported yet"),
               mlir::func::FuncOp();
      if (k.access == 0 || k.access == 2) // READ, RW
        for (size_t p = 0; p < k.points.size(); ++p) {
          inputs.push_back(k.elt);
          pointSlots.push_back({(int)i, p});
        }
      if (k.access == 1 || k.access == 2 || k.access == 3) { // WRITE, RW, INC
        if (outType && outType != k.elt)
          return fail("outputs of different element types"), mlir::func::FuncOp();
        outType = k.elt;
        ++numOutputs;
      }
      break;
    case KernelArgInfo::Kind::Gbl:
      if (k.access != 0)
        return fail("global argument that is written"), mlir::func::FuncOp();
      if (!k.elt)
        return fail("global of unsupported element type"), mlir::func::FuncOp();
      for (int e = 0; e < k.dim; ++e) {
        gblInputs.push_back(k.elt);
        gblSlots.push_back({(int)i, (size_t)e});
      }
      break;
    case KernelArgInfo::Kind::Idx:
      return fail("ops_arg_idx is not supported yet"), mlir::func::FuncOp();
    case KernelArgInfo::Kind::Reduce:
      return fail("reductions are not supported yet"), mlir::func::FuncOp();
    }
  }
  if (numOutputs == 0)
    return fail("kernel writes no dat"), mlir::func::FuncOp();

  inputs.insert(inputs.end(), gblInputs.begin(), gblInputs.end());
  for (const ConstUse &u : constUses)
    inputs.push_back(u.elt);

  std::vector<mlir::Type> results;
  if (numOutputs == 1) {
    results.push_back(outType);
  } else {
    inputs.push_back(mlir::MemRefType::get({numOutputs}, outType));
  }
  auto fn = mlir::func::FuncOp::create(loc, name, b.getFunctionType(inputs, results));
  fn.setPrivate();
  mlir::Block *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);

  pointValue.assign(args.size(), {});
  gblValue.assign(args.size(), {});
  size_t input = 0;
  for (auto [a, p] : pointSlots) {
    if (pointValue[a].empty())
      pointValue[a].resize(args[a].points.size());
    pointValue[a][p] = entry->getArgument(input++);
  }
  for (auto [a, e] : gblSlots) {
    if (gblValue[a].empty())
      gblValue[a].resize(args[a].dim);
    gblValue[a][e] = entry->getArgument(input++);
  }
  for (ConstUse &u : constUses) {
    u.value = entry->getArgument(input++);
    constRefList.push_back({u.var->getNameAsString(), u.offset, u.elt});
  }
  mlir::Value outMemref = numOutputs > 1 ? entry->getArgument(input) : mlir::Value();

  env.out.assign(args.size(), mlir::Value());
  for (size_t i = 0; i < args.size(); ++i) // an RW output starts from its current value
    if (args[i].kind == KernelArgInfo::Kind::Dat && args[i].access == 2) {
      for (size_t p = 0; p < args[i].points.size(); ++p)
        if (args[i].points[p] == std::array<int, 3>{0, 0, 0})
          env.out[i] = pointValue[i][p];
      if (!env.out[i])
        return fail("RW argument whose stencil lacks the centre point"), mlir::func::FuncOp();
    }

  emitStmt(decl->getBody());
  if (!ok_)
    return {};

  std::vector<mlir::Value> outs;
  for (size_t i = 0; i < args.size(); ++i) {
    const KernelArgInfo &k = args[i];
    if (k.kind == KernelArgInfo::Kind::Dat && (k.access == 1 || k.access == 2 || k.access == 3)) {
      if (!env.out[i])
        return fail("output argument " + llvm::Twine(i) + " is not assigned on every path"),
               mlir::func::FuncOp();
      outs.push_back(env.out[i]);
    }
  }
  if (numOutputs == 1) {
    b.create<mlir::func::ReturnOp>(loc, outs);
  } else {
    for (size_t i = 0; i < outs.size(); ++i) {
      mlir::Value idx = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(i));
      b.create<mlir::memref::StoreOp>(loc, outs[i], outMemref, mlir::ValueRange{idx});
    }
    b.create<mlir::func::ReturnOp>(loc);
  }
  return fn;
}

} // namespace

mlir::func::FuncOp KernelIRBuilder::generateAccessor(
    const std::vector<std::string> &sourceFiles, const std::string &kernelName,
    const std::string &functionName, int indexRank, const std::vector<KernelArgInfo> &args,
    const std::map<std::string, const void *> &constants, llvm::raw_ostream &errs,
    std::vector<KernelConstRef> *constRefs, const std::string &preamble) {
  for (const std::string &file : sourceFiles) {
    auto code = llvm::MemoryBuffer::getFile(file);
    if (!code)
      continue;
    // Parsing a kernel file with the application's headers costs about a second, and
    // every kernel in it, probed and then translated again for each module, would
    // pay that. The parse depends only on the file and the preamble, so keep it.
    static std::map<std::pair<std::string, std::string>, std::unique_ptr<clang::ASTUnit>> parsed;
    auto &unit = parsed[{file, preamble}];
    if (!unit) {
      std::string text = std::string(kPrelude) + preamble + "\n#line 1\n" + (*code)->getBuffer().str();
      unit = clang::tooling::buildASTFromCodeWithArgs(
          text, {"-x", "c++", "-std=c++17", "-ferror-limit=0", "-resource-dir=" OPS_CLANG_RESOURCE_DIR}, file);
    }
    if (!unit)
      continue;

    struct Finder : clang::RecursiveASTVisitor<Finder> {
      llvm::StringRef name;
      const clang::FunctionDecl *found = nullptr;
      bool VisitFunctionDecl(clang::FunctionDecl *d) {
        if (d->getDeclName().isIdentifier() && d->hasBody() && d->getName() == name)
          found = d;
        return true;
      }
    } finder;
    finder.name = kernelName;
    finder.TraverseDecl(unit->getASTContext().getTranslationUnitDecl());
    if (!finder.found)
      continue;

    // A body that did not parse cleanly (an undeclared name, say) has expressions the
    // translator would mis-read; refuse it rather than guess.
    struct Unresolved : clang::RecursiveASTVisitor<Unresolved> {
      bool found = false;
      bool VisitRecoveryExpr(clang::RecoveryExpr *) { return found = true, false; }
    } unresolved;
    unresolved.TraverseStmt(const_cast<clang::Stmt *>(finder.found->getBody()));
    if (unresolved.found || finder.found->isInvalidDecl()) {
      errs << "accessor kernel '" << kernelName
           << "': the body has unresolved names (see the compiler errors)\n";
      return {};
    }

    Translator tr(context_, finder.found, args, indexRank, constants, errs);
    mlir::func::FuncOp fn = tr.run(functionName);
    if (fn && constRefs)
      *constRefs = tr.constRefs();
    return fn;
  }
  errs << "accessor kernel '" << kernelName << "': definition not found in the registered sources\n";
  return {};
}

} // namespace ops_mlir

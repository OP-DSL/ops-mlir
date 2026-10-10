#include "ops-mlir/Runtime/PackedArgs.h"

#include <algorithm>
#include <cstring>

namespace ops_mlir::runtime {

size_t sizeOfElem(const std::string &elem) {
  if (elem == "f64" || elem == "i64") return 8;
  if (elem == "f32" || elem == "i32") return 4;
  if (elem == "f16" || elem == "bf16" || elem == "i16") return 2;
  return 0;
}

void PackedArgs::addMemref(void *data, const std::vector<int64_t> &sizes, const std::vector<int64_t> &strides) {
  Memref m;
  const size_t rank = sizes.size();
  m.words[0] = reinterpret_cast<int64_t>(data);
  m.words[1] = reinterpret_cast<int64_t>(data);
  m.words[2] = 0;
  for (size_t i = 0; i < rank; ++i) {
    m.words[3 + i] = sizes[i];
    m.words[3 + rank + i] = strides[i];
  }
  memrefs.push_back(m);
  memrefInfo.push_back({data, sizes, strides});
}

void PackedArgs::addScalar(const std::string &elem, const void *bytes) {
  ScalarInfo s;
  s.elem = elem;
  s.bytes.fill(0);
  std::memcpy(s.bytes.data(), bytes, sizeOfElem(elem));
  scalarInfo.push_back(std::move(s));
}

void PackedArgs::setResults(const std::vector<std::string> &elems) {
  resultElem = elems;
  resultOffset.clear();
  size_t offset = 0;
  for (const std::string &e : resultElem) {
    size_t size = sizeOfElem(e);
    offset = (offset + size - 1) / size * size;
    resultOffset.push_back(offset);
    offset += size;
  }
  results.assign(std::max<size_t>(offset, 16) + 16, 0);
}

void PackedArgs::finalize() {
  argv.clear();
  memrefPointers.clear();
  for (Memref &m : memrefs) memrefPointers.push_back(&m);
  // The storage of an argument is a pointer to the descriptor: args[i] -> (Memref *) -> descriptor.
  for (Memref *&p : memrefPointers) argv.push_back(&p);
  for (int64_t &v : indices) argv.push_back(&v);
  for (ScalarInfo &s : scalarInfo) argv.push_back(s.bytes.data());
  if (!resultElem.empty()) argv.push_back(results.data());
}

double PackedArgs::resultAsDouble(size_t i) const {
  const char *p = results.data() + resultOffset[i];
  if (resultElem[i] == "f64") {
    double v;
    std::memcpy(&v, p, 8);
    return v;
  }
  if (resultElem[i] == "f32") {
    float v;
    std::memcpy(&v, p, 4);
    return v;
  }
  return static_cast<double>(resultAsInt(i));
}

int64_t PackedArgs::resultAsInt(size_t i) const {
  const char *p = results.data() + resultOffset[i];
  if (resultElem[i] == "i32") {
    int32_t v;
    std::memcpy(&v, p, 4);
    return v;
  }
  if (resultElem[i] == "i64") {
    int64_t v;
    std::memcpy(&v, p, 8);
    return v;
  }
  return static_cast<int64_t>(resultAsDouble(i));
}

} // namespace ops_mlir::runtime

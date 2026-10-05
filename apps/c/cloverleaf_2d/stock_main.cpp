// The stock build of CloverLeaf: clover_leaf.cpp's main() is renamed for the ops-mlir
// build (a per-source property, so it applies to this target as well); forward to it.
int clover_original_main(int argc, char **argv);
int main(int argc, char **argv) { return clover_original_main(argc, argv); }

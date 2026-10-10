// Boundary condition exchange code on opensbliblock00 direction 0 left
ops_halo_group periodicBC_direction0_side0_33_block0 ;
{
int halo_iter[] = {2, block0np1 + 4, block0np2 + 4};
int from_base[] = {0, -2, -2};
int to_base[] = {block0np0, -2, -2};
int from_dir[] = {1, 2, 3};
int to_dir[] = {1, 2, 3};
ops_halo halo0 = ops_decl_halo(rho_B0, rho_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo1 = ops_decl_halo(rhou0_B0, rhou0_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo2 = ops_decl_halo(rhou1_B0, rhou1_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo3 = ops_decl_halo(rhou2_B0, rhou2_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo4 = ops_decl_halo(rhoE_B0, rhoE_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo grp[] = {halo0,halo1,halo2,halo3,halo4};
periodicBC_direction0_side0_33_block0 = ops_decl_halo_group(5,grp);
}
// Boundary condition exchange code on opensbliblock00 direction 0 right
ops_halo_group periodicBC_direction0_side1_34_block0 ;
{
int halo_iter[] = {2, block0np1 + 4, block0np2 + 4};
int from_base[] = {block0np0 - 2, -2, -2};
int to_base[] = {-2, -2, -2};
int from_dir[] = {1, 2, 3};
int to_dir[] = {1, 2, 3};
ops_halo halo0 = ops_decl_halo(rho_B0, rho_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo1 = ops_decl_halo(rhou0_B0, rhou0_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo2 = ops_decl_halo(rhou1_B0, rhou1_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo3 = ops_decl_halo(rhou2_B0, rhou2_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo4 = ops_decl_halo(rhoE_B0, rhoE_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo grp[] = {halo0,halo1,halo2,halo3,halo4};
periodicBC_direction0_side1_34_block0 = ops_decl_halo_group(5,grp);
}
// Boundary condition exchange code on opensbliblock00 direction 1 left
ops_halo_group periodicBC_direction1_side0_35_block0 ;
{
int halo_iter[] = {block0np0 + 4, 2, block0np2 + 4};
int from_base[] = {-2, 0, -2};
int to_base[] = {-2, block0np1, -2};
int from_dir[] = {1, 2, 3};
int to_dir[] = {1, 2, 3};
ops_halo halo0 = ops_decl_halo(rho_B0, rho_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo1 = ops_decl_halo(rhou0_B0, rhou0_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo2 = ops_decl_halo(rhou1_B0, rhou1_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo3 = ops_decl_halo(rhou2_B0, rhou2_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo4 = ops_decl_halo(rhoE_B0, rhoE_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo grp[] = {halo0,halo1,halo2,halo3,halo4};
periodicBC_direction1_side0_35_block0 = ops_decl_halo_group(5,grp);
}
// Boundary condition exchange code on opensbliblock00 direction 1 right
ops_halo_group periodicBC_direction1_side1_36_block0 ;
{
int halo_iter[] = {block0np0 + 4, 2, block0np2 + 4};
int from_base[] = {-2, block0np1 - 2, -2};
int to_base[] = {-2, -2, -2};
int from_dir[] = {1, 2, 3};
int to_dir[] = {1, 2, 3};
ops_halo halo0 = ops_decl_halo(rho_B0, rho_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo1 = ops_decl_halo(rhou0_B0, rhou0_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo2 = ops_decl_halo(rhou1_B0, rhou1_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo3 = ops_decl_halo(rhou2_B0, rhou2_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo4 = ops_decl_halo(rhoE_B0, rhoE_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo grp[] = {halo0,halo1,halo2,halo3,halo4};
periodicBC_direction1_side1_36_block0 = ops_decl_halo_group(5,grp);
}
// Boundary condition exchange code on opensbliblock00 direction 2 left
ops_halo_group periodicBC_direction2_side0_37_block0 ;
{
int halo_iter[] = {block0np0 + 4, block0np1 + 4, 2};
int from_base[] = {-2, -2, 0};
int to_base[] = {-2, -2, block0np2};
int from_dir[] = {1, 2, 3};
int to_dir[] = {1, 2, 3};
ops_halo halo0 = ops_decl_halo(rho_B0, rho_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo1 = ops_decl_halo(rhou0_B0, rhou0_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo2 = ops_decl_halo(rhou1_B0, rhou1_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo3 = ops_decl_halo(rhou2_B0, rhou2_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo4 = ops_decl_halo(rhoE_B0, rhoE_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo grp[] = {halo0,halo1,halo2,halo3,halo4};
periodicBC_direction2_side0_37_block0 = ops_decl_halo_group(5,grp);
}
// Boundary condition exchange code on opensbliblock00 direction 2 right
ops_halo_group periodicBC_direction2_side1_38_block0 ;
{
int halo_iter[] = {block0np0 + 4, block0np1 + 4, 2};
int from_base[] = {-2, -2, block0np2 - 2};
int to_base[] = {-2, -2, -2};
int from_dir[] = {1, 2, 3};
int to_dir[] = {1, 2, 3};
ops_halo halo0 = ops_decl_halo(rho_B0, rho_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo1 = ops_decl_halo(rhou0_B0, rhou0_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo2 = ops_decl_halo(rhou1_B0, rhou1_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo3 = ops_decl_halo(rhou2_B0, rhou2_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo halo4 = ops_decl_halo(rhoE_B0, rhoE_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo grp[] = {halo0,halo1,halo2,halo3,halo4};
periodicBC_direction2_side1_38_block0 = ops_decl_halo_group(5,grp);
}

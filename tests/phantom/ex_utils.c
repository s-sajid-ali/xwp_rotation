static char help[] = "Rotate a 3D object stored as 2D matrix!\n\n";

/* Utilities for rotation of 3D object stored as 2D matrix! */

#include <petscmat.h>
#include <petscviewerhdf5.h>

#ifndef STRUCT_APPCTX
#define STRUCT_APPCTX

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Mat            beta_in;              /* beta matrix*/
  Mat            beta_out;             /* rotated beta matrix*/
  Mat            rot;                  /* rotation matrix*/
  Vec            X;                    /* work vector */
  Vec            Y;                    /* work vector */  
  PetscInt       cols;                 /* object size is 64^3 */
  PetscMPIInt    rank,size;
  PetscBool      debug_flag;
} AppCtx;

#endif


/* --------------------------------------------------------------------- 
   initialize - Initialize the problem.
   AppCtx - user-defined application context  
 --------------------------------------------------------------------- */
PetscErrorCode initialize(void *ctx)
{
    PetscErrorCode ierr;
    AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
    
    appctx->debug_flag  = PETSC_TRUE;
    appctx->cols        = 64;
    PetscInt _cols = appctx->cols;
    
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&appctx->rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&appctx->size);CHKERRQ(ierr);

    /* Create (sparse) rotation matrix */
    ierr = MatCreate(PETSC_COMM_WORLD,&appctx->rot);CHKERRQ(ierr);
    ierr = MatSetType(appctx->rot,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(appctx->rot,PETSC_DECIDE,PETSC_DECIDE,_cols*_cols,_cols*_cols);CHKERRQ(ierr);
    ierr = MatSetFromOptions(appctx->rot);CHKERRQ(ierr);
    
    /* Create (dense) refractive index matrix */
    ierr = MatCreate(PETSC_COMM_WORLD,&appctx->beta_in);CHKERRQ(ierr);
    ierr = MatSetType(appctx->beta_in,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetSizes(appctx->beta_in,PETSC_DECIDE,PETSC_DECIDE,_cols*_cols,_cols);CHKERRQ(ierr);
    ierr = MatSetFromOptions(appctx->beta_in);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&appctx->beta_out);CHKERRQ(ierr);
    ierr = MatSetType(appctx->beta_out,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetSizes(appctx->beta_out,PETSC_DECIDE,PETSC_DECIDE,_cols*_cols,_cols);CHKERRQ(ierr);
    ierr = MatSetFromOptions(appctx->beta_out);CHKERRQ(ierr);
    ierr = MatSetUp(appctx->beta_out); CHKERRQ(ierr);

    /* Create work vectors */
    ierr = VecCreateMPI(PETSC_COMM_WORLD,((_cols*_cols)/(appctx->size)),PETSC_DECIDE,&appctx->X); CHKERRQ(ierr);
    ierr = VecSetBlockSize(appctx->X, _cols); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx->X, "X_vec");CHKERRQ(ierr);
    ierr = VecCreateMPI(PETSC_COMM_WORLD,((_cols*_cols)/(appctx->size)),PETSC_DECIDE,&appctx->Y); CHKERRQ(ierr);
    ierr = VecSetBlockSize(appctx->X, _cols); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx->Y, "Y_vec");CHKERRQ(ierr);

    /* Load refractive indices */
    /*remove beta_out_view after debugging!*/
    PetscViewer    beta_in_view;
    PetscViewer    beta_out_view;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                                 "beta.dat",\
                                 FILE_MODE_READ,\
                                 &beta_in_view);CHKERRQ(ierr);
    ierr = MatLoad(appctx->beta_in,beta_in_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&beta_in_view);CHKERRQ(ierr);
    
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                                 "beta.dat",\
                                 FILE_MODE_READ,\
                                 &beta_out_view);CHKERRQ(ierr);
    ierr = MatLoad(appctx->beta_out,beta_out_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&beta_out_view);CHKERRQ(ierr);
    
    /* Load precomputed rotation matrix */
    PetscViewer    rot_view;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                                 "rot.dat",\
                                 FILE_MODE_READ,\
                                 &rot_view);CHKERRQ(ierr);
    ierr = MatLoad(appctx->rot,rot_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&rot_view);CHKERRQ(ierr);

  return ierr;
}


/* --------------------------------------------------------------------- 
   finalize - finalze the problem.
   AppCtx - user-defined application context  
 --------------------------------------------------------------------- */
PetscErrorCode finalize(void *ctx)
{
    PetscErrorCode ierr;
    AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
    
    ierr = MatDestroy(&appctx->beta_in); CHKERRQ(ierr);
    ierr = MatDestroy(&appctx->beta_out); CHKERRQ(ierr);
    ierr = MatDestroy(&appctx->rot); CHKERRQ(ierr);
    ierr = VecDestroy(&appctx->X); CHKERRQ(ierr);
    ierr = VecDestroy(&appctx->Y); CHKERRQ(ierr);
    
  return ierr;
}


/* --------------------------------------------------------------------- 
   submatrixtovector - convert (sub)matrix to 1D vector, perform MatMult ! .
   AppCtx - user-defined application context  
 --------------------------------------------------------------------- */
PetscErrorCode submatrixtovector(void *ctx)
{
    PetscErrorCode ierr;
    AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
    
    PetscInt           matstart,matend;
    PetscInt           row,col;
    PetscInt           rowidx[1];
    PetscInt           colidx[appctx->cols];
    const PetscScalar  *data;

    for(col=0; col<appctx->cols; col++){colidx[col] = col;}

    ierr = MatGetOwnershipRange(appctx->beta_in,&matstart,&matend); CHKERRQ(ierr);

    for (row=matstart; row<appctx->cols; row++){
        rowidx[0] = row;
        ierr = MatGetRow(appctx->beta_in,row,NULL,NULL,&data); CHKERRQ(ierr);
        ierr = VecSetValuesBlocked(appctx->X,1, rowidx, &data[0], INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatRestoreRow(appctx->beta_in,row,NULL,NULL,&data); CHKERRQ(ierr);
    }

    ierr = VecAssemblyBegin(appctx->X); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(appctx->X); CHKERRQ(ierr);
    
    if (appctx->debug_flag){
    ierr = VecSetBlockSize(appctx->X,1); CHKERRQ(ierr);
    PetscViewer x_view;  /* viewer to write the solution to hdf5*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"X.h5",
                               FILE_MODE_WRITE,&x_view);CHKERRQ(ierr);
    ierr = VecView(appctx->X,x_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&x_view);CHKERRQ(ierr);        
    }
    
  return ierr;
}


/* --------------------------------------------------------------------- 
   rotate - rotate using discretized rotation matrix
   AppCtx - user-defined application context  
 --------------------------------------------------------------------- */
PetscErrorCode rotate(void *ctx)
{
    PetscErrorCode ierr;
    AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
    
    /* Matrix vector multiplication */
    ierr = MatMult(appctx->rot,appctx->X,appctx->Y); CHKERRQ(ierr);

    
    if (appctx->debug_flag){
    ierr = VecSetBlockSize(appctx->Y,1); CHKERRQ(ierr);
    PetscViewer y_view;  /* viewer to write the solution to hdf5*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"Y.h5",
                               FILE_MODE_WRITE,&y_view);CHKERRQ(ierr);
    ierr = VecView(appctx->Y,y_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&y_view);CHKERRQ(ierr);
    }
    

  return ierr;
}


/* --------------------------------------------------------------------- 
   rotate - rotate using discretized rotation matrix
   AppCtx - user-defined application context  
 --------------------------------------------------------------------- */
PetscErrorCode vectortosubmatrix(void *ctx)
{
    PetscErrorCode ierr;
    AppCtx             *appctx = (AppCtx*) ctx;   /* user-defined application context */
    PetscInt           vecstart,vecend,element;
    PetscInt           rowstart,rowend;
    PetscInt           row,col,rowlocal;
    PetscInt           rowidx[1];
    PetscInt           colidx[appctx->cols];
    const PetscScalar  *_data;
    
    for(col=0; col<appctx->cols; col++){colidx[col] = col;}
    
    /* convert 1D vector to (sub)matrix*/  
    ierr = VecGetOwnershipRange(appctx->Y,&vecstart,&vecend); CHKERRQ(ierr);
    ierr = VecGetArrayRead(appctx->Y, &_data); CHKERRQ(ierr);
    rowstart = vecstart/appctx->cols;
    rowend   = vecend/appctx->cols;
    
    PetscPrintf(PETSC_COMM_SELF,"rank: %d, rowstart: %d, rowend: %d\n", appctx->rank,rowstart,rowend);
    

    for (row=rowstart; row<rowend; row++){
        rowidx[0] = row;
        MatSetValues(appctx->beta_in, 1, rowidx, appctx->cols, colidx,\
                     (_data + (row-rowstart)*appctx->cols), INSERT_VALUES); CHKERRQ(ierr);
    }
    
    ierr = MatAssemblyBegin(appctx->beta_in,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(appctx->beta_in,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(appctx->Y, &_data); CHKERRQ(ierr);
    
    if (appctx->debug_flag){
    ierr = VecSetBlockSize(appctx->Y,1); CHKERRQ(ierr);
    PetscViewer y_view;  /* viewer to write the solution to hdf5*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"Y.h5",
                               FILE_MODE_WRITE,&y_view);CHKERRQ(ierr);
    ierr = VecView(appctx->Y,y_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&y_view);CHKERRQ(ierr);
    }
    
  return ierr;
}

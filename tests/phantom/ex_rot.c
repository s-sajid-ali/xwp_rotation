static char help[] = "Rotate a 3D object stored as 2D matrix!\n\n";

/* Rotate a 3D object stored as 2D matrix! */

#include <petscmat.h>
#include <petscviewerhdf5.h>

int main(int argc,char **args)
{
  Mat            beta;                      /* beta matrix*/
  Mat            rot;                       /* rotation matrix*/
  Vec            X;                         /* work vector */
  Vec            Y;                         /* work vector */
  char           file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscInt       cols = 64;                 /* object size is 64^3 */ 

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  
  /* Create matrix and vector*/
  ierr = MatCreate(PETSC_COMM_WORLD,&rot);CHKERRQ(ierr);
  ierr = MatSetType(rot,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(rot,PETSC_DECIDE,PETSC_DECIDE,cols*cols,cols*cols);CHKERRQ(ierr);
  ierr = MatSetFromOptions(rot);CHKERRQ(ierr);
  /* Open binary file. */ 
  PetscViewer    rot_view;  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                               "rot.dat",\
                               FILE_MODE_READ,\
                               &rot_view);CHKERRQ(ierr);  
  ierr = MatLoad(rot,rot_view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&rot_view);CHKERRQ(ierr);
    
    
  /* Create matrix and vector*/
  ierr = MatCreate(PETSC_COMM_WORLD,&beta);CHKERRQ(ierr);
  ierr = MatSetType(beta,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetSizes(beta,PETSC_DECIDE,PETSC_DECIDE,cols*cols,cols);CHKERRQ(ierr);
  ierr = MatSetFromOptions(beta);CHKERRQ(ierr);
  /* Open binary file. */ 
  PetscViewer    beta_view;  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                               "beta.dat",\
                               FILE_MODE_READ,\
                               &beta_view);CHKERRQ(ierr);  
  ierr = MatLoad(beta,beta_view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&beta_view);CHKERRQ(ierr);
    
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,cols*cols,&X); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X, "X_vec");CHKERRQ(ierr);    
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,cols*cols,&Y); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Y, "Y_vec");CHKERRQ(ierr);    

    
  /* convert matrix to 1D vector! */
  PetscInt           istart,iend,row,col;
  PetscInt           idx[1];
  const PetscScalar  *data;
  
  ierr = VecSetBlockSize(X, cols); CHKERRQ(ierr);
    
  ierr = MatGetOwnershipRange(beta,&istart,&iend); CHKERRQ(ierr);

  for (row=istart; row<cols; row++){
      idx[0] = row;
      ierr = MatGetRow(beta,row,NULL,NULL,&data); CHKERRQ(ierr);
      ierr = VecSetValuesBlocked(X,1, idx, data, INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatRestoreRow(beta,row,NULL,NULL,&data); CHKERRQ(ierr);
  }
    
  ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X); CHKERRQ(ierr);

  // Matrix vector multiplication
  ierr = MatMult(rot,X,Y); CHKERRQ(ierr);
  
  ierr = VecSetBlockSize(X,1); CHKERRQ(ierr);

  PetscViewer x_view;  /* viewer to write the solution to hdf5*/ 
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"X.h5",
                             FILE_MODE_WRITE,&x_view);CHKERRQ(ierr);
  ierr = VecView(X,x_view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&x_view);CHKERRQ(ierr);
  
    
  PetscViewer y_view; 
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"Y.h5",
                             FILE_MODE_WRITE,&y_view);CHKERRQ(ierr);
  ierr = VecView(Y,y_view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&y_view);CHKERRQ(ierr);
  
    
  ierr = MatDestroy(&beta); CHKERRQ(ierr);
  ierr = MatDestroy(&rot); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&Y); CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

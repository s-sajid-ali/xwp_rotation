static char help[] = "Rotate a 3D object stored as 2D matrix!\n\n";

/* Rotate a 3D object stored as 2D matrix! */

#include <petscmat.h>
#include <petscviewerhdf5.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Mat            beta;                      /* beta matrix*/
  Mat            _beta;                     /* rotated beta matrix*/
  Mat            rot;                       /* rotation matrix*/
  Vec            X;                         /* work vector */
  Vec            Y;                         /* work vector */  
  PetscInt       cols = 64;                 /* object size is 64^3 */
} AppCtx;

//extern PetscErrorCode submatrixtovector(Mat, Vec, PetscInt);
//extern PetscErrorCode rotate(Mat, Vec, Vec, PetscInt);
//extern PetscErrorCode vectortosubmatrix(Mat, Vec, PetscInt);

int main(int argc,char **args)
{
  char           file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscMPIInt    rank,size;

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  
  /* Create matrix and vector*/
  ierr = MatCreate(PETSC_COMM_WORLD,&appctx.rot);CHKERRQ(ierr);
  ierr = MatSetType(appctx.rot,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(appctx.rot,PETSC_DECIDE,PETSC_DECIDE,appctx.cols*appctx.cols,appctx.cols*appctx.cols);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx.rot);CHKERRQ(ierr);

  /* Open binary file. */ 
  PetscViewer    rot_view;  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                               "rot.dat",\
                               FILE_MODE_READ,\
                               &rot_view);CHKERRQ(ierr);  
  ierr = MatLoad(appctx.rot,rot_view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&rot_view);CHKERRQ(ierr);
    
    
  /* Create matrices/vectors */
  ierr = MatCreate(PETSC_COMM_WORLD,&appctx.beta);CHKERRQ(ierr);
  ierr = MatSetType(appctx.beta,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetSizes(appctx.beta,PETSC_DECIDE,PETSC_DECIDE,appctx.cols*appctx.cols,appctx.cols);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx.beta);CHKERRQ(ierr);
    
  ierr = MatCreate(PETSC_COMM_WORLD,&appctx._beta);CHKERRQ(ierr);
  ierr = MatSetType(appctx._beta,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetSizes(appctx._beta,PETSC_DECIDE,PETSC_DECIDE,appctx.cols*appctx.cols,appctx.cols);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx._beta);CHKERRQ(ierr);
  ierr = MatSetUp(appctx._beta); CHKERRQ(ierr);

  ierr = VecCreateMPI(PETSC_COMM_WORLD,((appctx.cols*appctx.cols)/(size)),PETSC_DECIDE,&appctx.X); CHKERRQ(ierr);
  ierr = VecSetBlockSize(appctx.X, appctx.cols); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.X, "X");CHKERRQ(ierr);    
  ierr = VecCreateMPI(PETSC_COMM_WORLD,((appctx.cols*appctx.cols)/(size)),PETSC_DECIDE,&appctx.Y); CHKERRQ(ierr);
  ierr = VecSetBlockSize(appctx.X, appctx.cols); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.Y, "Y");CHKERRQ(ierr);    

  /* Open binary file. */ 
  PetscViewer    beta_view;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                               "beta.dat",\
                               FILE_MODE_READ,\
                               &beta_view);CHKERRQ(ierr);  
  ierr = MatLoad(appctx.beta,beta_view);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                               "beta.dat",\
                               FILE_MODE_READ,\
                               &beta_view);CHKERRQ(ierr);  
  ierr = MatLoad(appctx._beta,beta_view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&beta_view);CHKERRQ(ierr);
    
  /* convert (sub)matrix to 1D vector, perform MatMult ! */
  PetscInt           matstart,matend;
  PetscInt           vecstart,vecend,element;
  PetscInt           rowstart,rowend;
  PetscInt           row,col;
  PetscInt           rowidx[1];
  PetscInt           colidx[cols];
  const PetscScalar  *_data;
    
  for(col=0; col<cols; col++){colidx[col] = col;}

  ierr = MatGetOwnershipRange(beta,&matstart,&matend); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "rank : %d, mat own : %d : %d ! \n", rank, matstart,matend);    

  for (row=matstart; row<cols; row++){
      rowidx[0] = row;
      ierr = MatGetRow(beta,row,NULL,NULL,&data); CHKERRQ(ierr);
      ierr = VecSetValuesBlocked(X,1, rowidx, &data[0], INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatRestoreRow(beta,row,NULL,NULL,&data); CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
  
  /* Matrix vector multiplication */
  ierr = MatMult(rot,X,Y); CHKERRQ(ierr);
  
  /* convert 1D vector to (sub)matrix*/  
  ierr = VecGetOwnershipRange(Y,&vecstart,&vecend); CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y, &_data); CHKERRQ(ierr);
  rowstart = vecstart/cols;
  rowend   = vecend/cols;
    
  PetscScalar *_ref;
  _ref = &_data;  

  PetscPrintf(PETSC_COMM_SELF,"\n rank: %d, %g+%gi\n",rank,(double)PetscRealPart(_ref[10]),(double)PetscImaginaryPart(_ref[10]));


  for (row=rowstart; row<rowend; row++){
      rowidx[0] = row;
      _ref = &_data[row*cols];
      MatSetValues(_beta, 1, rowidx, cols, colidx, _ref, INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(_beta,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(_beta,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Y, &_data); CHKERRQ(ierr);
  
  ierr = VecSetBlockSize(Y,1); CHKERRQ(ierr);  
  PetscViewer y_view; 
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"Y_rot.h5",
                             FILE_MODE_WRITE,&y_view);CHKERRQ(ierr);
  ierr = VecView(Y,y_view); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&y_view); CHKERRQ(ierr);
       
    
  ierr = MatGetOwnershipRange(_beta,&matstart,&matend); CHKERRQ(ierr);
  for (row=matstart; row<cols; row++){
      rowidx[0] = row;
      ierr = MatGetRow(_beta,row,NULL,NULL,&data); CHKERRQ(ierr);
      ierr = VecSetValuesBlocked(X,1, rowidx, data, INSERT_VALUES); CHKERRQ(ierr);
      ierr = MatRestoreRow(_beta,row,NULL,NULL,&data); CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X); CHKERRQ(ierr);
  ierr = VecSetBlockSize(X,1); CHKERRQ(ierr);
  PetscViewer x_view;  /* viewer to write the solution to hdf5*/ 
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"X_rot.h5",
                             FILE_MODE_WRITE,&x_view);CHKERRQ(ierr);
  ierr = VecView(X,x_view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&x_view);CHKERRQ(ierr);
  
  
  //ierr = MatView(_beta,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  
    
  ierr = MatDestroy(&beta); CHKERRQ(ierr);
  ierr = MatDestroy(&_beta); CHKERRQ(ierr);
  ierr = MatDestroy(&rot); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&Y); CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static char help[] = "Rotate a 3D object stored as 2D matrix!\n\n";

/* Rotate a 3D object stored as 2D matrix! */

#include <petscmat.h>
#include <petscviewerhdf5.h>



extern PetscErrorCode initialize(void*);
extern PetscErrorCode finalize(void*);
extern PetscErrorCode submatovector(Mat, Vec, PetscInt, char *, void *);
extern PetscErrorCode rotate(Mat, Vec, Vec, char *, void*);
extern PetscErrorCode vectortosubmatrix(Vec, Mat, PetscInt, char *, void *);


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


int main(int argc,char **args)
{
    AppCtx           appctx;          /* user-defined application context */
    PetscErrorCode   ierr;
    char fname[50];
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    
    initialize(&appctx);
	
    sprintf(fname,"X.h5");
    submatovector(appctx.beta_in, appctx.X, 0, fname, &appctx);

    sprintf(fname,"Y_rot.h5");
    rotate(appctx.rot, appctx.X, appctx.Y, fname, &appctx);

    sprintf(fname,"Y.h5");
    vectortosubmatrix(appctx.Y, appctx.beta_out, 0, fname, &appctx);

    sprintf(fname,"X_rot.h5");
    submatovector(appctx.beta_out, appctx.X, 0, fname, &appctx);
    
    finalize(&appctx);
    
    ierr = PetscFinalize();
    return ierr;
}

static char help[] = "Rotate a 3D object stored as 2D matrix!\n\n";

/* Rotate a 3D object stored as 2D matrix! */

#include <petscmat.h>
#include <petscviewerhdf5.h>



extern PetscErrorCode initialize(void*);
extern PetscErrorCode finalize(void*);
extern PetscErrorCode submatovector(Mat, Vec, PetscInt, void *ctx);
extern PetscErrorCode rotate(Mat, Vec, Vec, void*);
extern PetscErrorCode vectortosubmatrix(Vec, Mat, PetscInt, void *);


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
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    
    initialize(&appctx);

    submatovector(appctx.beta_in, appctx.X, 0, &appctx);
    rotate(appctx.rot, appctx.X, appctx.Y, &appctx);
    vectortosubmatrix(appctx.Y, appctx.beta_out, 0, &appctx);
    submatovector(appctx.beta_out, appctx.X, 0, &appctx);
    
    finalize(&appctx);
    
    ierr = PetscFinalize();
    return ierr;
}

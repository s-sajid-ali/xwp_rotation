CFLAGS     =
CPPFLAGS   =
LIBFILES   =
TARGET     = ex_full_object
OBJFILES   = ex_full_object.o
CLEANFILES = $(TARGET)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

all: $(TARGET)

$(TARGET) : $(OBJFILES)
	${CLINKER} -o $(TARGET) $(OBJFILES) ${PETSC_KSP_LIB}

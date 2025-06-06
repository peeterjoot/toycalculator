program main
    implicit none
    integer :: argc
    character(len=32), allocatable :: argv(:)

    ! Get the number of command-line arguments
    argc = command_argument_count()

    if (argc > 0) then
        write(*, '(A)') 'hi'
    else
        write(*, '(A)') 'bye'
    end if

    stop 0
end program main

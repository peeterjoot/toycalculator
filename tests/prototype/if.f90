program ifscope
    implicit none
    integer :: x
    integer(8) :: myScopeVar

    x = 3

    if ( x < 4 ) then
        myScopeVar = 1 + x
        print *, myScopeVar
    end if

    stop x
end program ifscope

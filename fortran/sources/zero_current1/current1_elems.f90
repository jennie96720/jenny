!Integration grids and kinematics for the nucleon currents (= qdq/current.py, qdq/exchange.py)
module current1_elems
    use libraries
    implicit none

type current_para
    integer:: nk = 32, nz = 16, ny = 12                 !one-loop grid (|k|, z, y)
    integer:: nko(3) = [12,6,6]                         !two-loop outer grid (|k|, z, y)
    integer:: nli(4) = [12,6,6,6]                       !two-loop inner grid (|l|, z, y, phi)
    real(8):: scale = 0.6d0
    real(8):: eta = 1d0/3d0                             !momentum fraction of the bystander quark
    integer:: nQ2 = 3                                   !number of Q^2 points
    real(8):: Q2(64) = 0d0                              ![GeV^2], first nQ2 entries used
    character(len=40):: root = 'current'
end type current_para

type current_grid
    integer:: n
    complex(8),allocatable:: k(:,:)                     !(4,n)
    real(8),allocatable:: w(:)                          !weights incl. 2pi/(2pi)^4 (or phi weight)
end type current_grid

public

contains

    !k = |k| (s_z s_y cos phi, s_z s_y sin phi, s_z y, z);  nphi = 0: phi = 0 and factor 2 pi
    function make_grid(nr,nz,ny,nphi,scale) result(gr)
        integer:: nr,nz,ny,nphi
        real(8):: scale
        type(current_grid):: gr
        real(8):: r(nr),wr(nr),z(nz),wz(nz),y(ny),wy(ny),sz,sy,ph,wph
        integer:: i,j,l,m,ip,np
        call radial(nr,scale,r,wr)
        call chebyshev_u(nz,z,wz)
        call gauleg(ny,y,wy)
        np = max(nphi,1)
        gr%n = nr*nz*ny*np
        allocate(gr%k(4,gr%n),gr%w(gr%n))
        ip = 0
        do i=1,nr; do j=1,nz; do l=1,ny; do m=1,np
            ip = ip+1
            sz = sqrt(1-z(j)**2); sy = sqrt(1-y(l)**2)
            if(nphi==0)then
                ph = 0d0; wph = 2*pi
            else
                ph = 2*pi*(m-0.5d0)/nphi; wph = 2*pi/nphi
            end if
            gr%k(:,ip) = r(i)*[sz*sy*cos(ph),sz*sy*sin(ph),sz*y(l),z(j)]
            gr%w(ip) = wr(i)*r(i)**3*wz(j)*wy(l)*wph/(2*pi)**4
        end do; end do; end do; end do
    end function

    !Breit frame: P_{f,i} = (0,0,0,iE) +- Q/2, Q = (0,0,|Q|,0)
    subroutine kinematics(M,Q,Pf,Pi,Qv)
        real(8):: M,Q
        complex(8):: Pf(4),Pi(4),Qv(4),Pbf(4)
        Pbf = [Zero,Zero,Zero,Im*sqrt(M**2+Q**2/4)]
        Qv = [Zero,Zero,dcmplx(Q,0d0),Zero]
        Pf = Pbf+Qv/2; Pi = Pbf-Qv/2
    end subroutine

end module

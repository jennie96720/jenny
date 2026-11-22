!Diquark axial couplings from the quark loop inside the diquark (= qdq/pdf_pol.py axial_triangle,
!axial_couplings_at, AxialCouplingTable)
!
!  T_a = int Tr[Gbar^X(q;-K) Shat_mu(q_a) G^Y(q;K) S^T(q_b)],  T_b = int Tr[Gbar^X S(q_a) G^Y Shat_mu(q_b)^T]
!  q_a = q + K/2, q_b = -q + K/2, Shat = unamputated forward axial vertex, colour trace 2.
!  Amputated Q = 0 vertices per unit flavour weight (sign convention of qdq/current.py):
!    1+1+ : -(T_a + T_b)/2 = c_aa eps_{mu al be nu} K_nu
!    0+1+ : -(T_a - T_b)/2 = c1 delta_{mu be} + c2 K_mu K_be / K^2   (and 1+0+)
!  The couplings depend on K^2 only; they fall steeply off-shell, so the nucleon diagrams use them at
!  each diquark's (complex) virtuality via a table in the upper half plane, c(z*) = -conj(c(z)).
!  Keys: 1 aa, 2 c1_sa, 3 c2_sa, 4 c1_as, 5 c2_as.
module axial_elems
    use libraries
    use quark_elems
    use diquark_elems
    use current1_elems
    use pdf_elems
    implicit none

type axial_table
    integer:: nre = 0, nim = 0
    real(8),allocatable:: re(:),im(:)
    complex(8),allocatable:: v(:,:,:)                   !(nre,nim,5)
    complex(8),allocatable:: yim(:,:),mim(:,:)          !splines along Im: (nim, nre*5)
contains
    procedure,public:: setup => axial_table_setup
    procedure,public:: eval  => axial_table_eval
    procedure,public:: read  => axial_table_read
    procedure,public:: write => axial_table_write
end type axial_table

public

contains

    subroutine axial_triangle(X,Y,K,g,gr,Ta,Tb)
        integer,intent(in):: X,Y
        complex(8),intent(in):: K(4),g(2)
        type(current_grid),intent(in):: gr
        complex(8),intent(out):: Ta(4,4,4),Tb(4,4,4)          !(mu,i,j)
        complex(8):: q(4),qa(4),qb(4),Fx,Fy,Gx(4,4,4),Gy(4,4,4),insa(4,4,4),insb(4,4,4),SqbT(4,4),Sqa(4,4)
        complex(8):: A1(4,4),B1(4,4)
        integer:: ip,mu,i,j,nx,ny
        real(8):: w
        nx = merge(1,4,X==1); ny = merge(1,4,Y==1)
        Ta = Zero; Tb = Zero
        do ip=1,gr%n
            if(.not.MPI_distribute(ip)) cycle
            q = gr%k(:,ip); w = gr%w(ip)
            qa = q+K/2; qb = -q+K/2
            Fx = g(X)*Famp(merge('s','a',X==1),dot4(q,q))
            Fy = g(Y)*Famp(merge('s','a',Y==1),dot4(q,q))
            if(X==1)then
                Gx(:,:,1) = Fx*bar_s
            else
                Gx = Fx*bar_a
            end if
            if(Y==1)then
                Gy(:,:,1) = Fy*gam_s
            else
                Gy = Fy*gam_a
            end if
            insa = axial_quark(qa); insb = axial_quark(qb)
            SqbT = transpose(propagator(qb)); Sqa = propagator(qa)
            do i=1,nx; do j=1,ny
                A1 = matmul(Gy(:,:,j),SqbT)
                B1 = matmul(Gx(:,:,i),matmul(Sqa,Gy(:,:,j)))
                do mu=1,4
                    Ta(mu,i,j) = Ta(mu,i,j)+2*w*tr2(matmul(Gx(:,:,i),insa(:,:,mu)),A1)
                    Tb(mu,i,j) = Tb(mu,i,j)+2*w*tr2(B1,transpose(insb(:,:,mu)))
                end do
            end do; end do
        end do
        do i=1,4
            call MPI_reducesum(Ta(:,:,i)); call MPI_reducesum(Tb(:,:,i))
        end do
    end subroutine

    !c(1:5) at complex K^2
    function axial_couplings_at(K2,g,gr) result(c)
        complex(8),intent(in):: K2,g(2)
        type(current_grid),intent(in):: gr
        complex(8):: c(5),K(4),kap,Ta(4,4,4),Tb(4,4,4),V(4,4,4),E(4,4,4),V2(4,4),PT(4,4),KK(4,4),KK2
        integer:: mu,a,b,n,XY
        kap = sqrt(-K2)
        K = [Zero,Zero,Zero,Im*kap]
        if(abs(K2)<1d-8) K = [Zero,Zero,dcmplx(1d-4,0d0),Zero]
        call axial_triangle(2,2,K,g,gr,Ta,Tb)
        V = -(Ta+Tb)/2
        do mu=1,4; do a=1,4; do b=1,4
            E(mu,a,b) = sum(eps4(mu,a,b,:)*K)
        end do; end do; end do
        c(1) = sum(V*E)/sum(E*E)
        KK2 = dot4(K,K)
        do a=1,4; do b=1,4
            KK(a,b) = K(a)*K(b)/KK2
            PT(a,b) = merge(One,Zero,a==b)-KK(a,b)
        end do; end do
        do XY=1,2
            if(XY==1)then
                call axial_triangle(1,2,K,g,gr,Ta,Tb)
                V2 = -(Ta(:,1,:)-Tb(:,1,:))/2
            else
                call axial_triangle(2,1,K,g,gr,Ta,Tb)
                V2 = -(Ta(:,:,1)-Tb(:,:,1))/2
            end if
            c(2*XY) = sum(V2*PT)/3
            c(2*XY+1) = sum(V2*KK)-c(2*XY)
        end do
    end function

    subroutine axial_table_setup(this)
        class(axial_table):: this
        integer:: j,key
        if(allocated(this%yim)) deallocate(this%yim,this%mim)
        allocate(this%yim(this%nim,this%nre*5),this%mim(this%nim,this%nre*5))
        do key=1,5; do j=1,this%nre
            this%yim(:,j+this%nre*(key-1)) = this%v(j,:,key)
        end do; end do
        call spline_setup(this%nim,this%im,this%nre*5,this%yim,this%mim)
    end subroutine

    !bicubic (not-a-knot) interpolation; outside the grid: nearest edge; lower half: -conj
    function axial_table_eval(this,K2) result(c)
        class(axial_table):: this
        complex(8),intent(in):: K2
        complex(8):: c(5),vals(this%nre*5),yre(this%nre,5),mre(this%nre,5)
        real(8):: x,y
        integer:: key
        x = min(max(real(K2),this%re(1)),this%re(this%nre))
        y = min(max(abs(aimag(K2)),this%im(1)),this%im(this%nim))
        vals = spline_eval(this%nim,this%im,this%nre*5,this%yim,this%mim,y)
        do key=1,5
            yre(:,key) = vals(1+this%nre*(key-1):this%nre*key)
        end do
        call spline_setup(this%nre,this%re,5,yre,mre)
        c = spline_eval(this%nre,this%re,5,yre,mre,x)
        if(aimag(K2)<0d0) c = -conjg(c)
    end function

    subroutine axial_table_write(this,fname,info)
        class(axial_table):: this
        character(len=*):: fname,info
        integer:: i,j
        if(jrank/=0) return
        open(unit=103,file=trim(fname),status='replace')
        write(103,'(a,a)') '# diquark axial couplings vs K^2 (Re >= ..., Im >= 0): ',trim(info)
        write(103,'(2i6)') this%nre,this%nim
        write(103,'(32f12.6)') this%re
        write(103,'(32f12.6)') this%im
        do i=1,this%nre; do j=1,this%nim
            write(103,'(2i4,10es20.11)') i,j,this%v(i,j,:)
        end do; end do
        close(103)
    end subroutine

    subroutine axial_table_read(this,fname)
        class(axial_table):: this
        character(len=*):: fname
        character(len=200):: line
        integer:: i,j,ii,jj
        real(8):: t(10)
        open(unit=103,file=trim(fname),status='old')
        read(103,'(a)') line
        read(103,*) this%nre,this%nim
        if(allocated(this%re)) deallocate(this%re,this%im,this%v)
        allocate(this%re(this%nre),this%im(this%nim),this%v(this%nre,this%nim,5))
        read(103,*) this%re
        read(103,*) this%im
        do i=1,this%nre; do j=1,this%nim
            read(103,*) ii,jj,t                            !list-directed: complex as (re, im) pairs of reals
            this%v(ii,jj,:) = dcmplx(t(1:9:2),t(2:10:2))
        end do; end do
        close(103)
        call this%setup()
    end subroutine

end module

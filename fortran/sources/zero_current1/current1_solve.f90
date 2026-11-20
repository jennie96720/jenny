!One-loop diagrams of the nucleon current (= qdq/current.py NucleonCurrent.diagrams)
!   1: probe on the bystander quark;  2: elastic probe-diquark;  3: 0+ <-> 1+ transition
!   J_m = int_k Psibar(p;-P_f) V_m Psi(k;P_i),  projected point by point on Lorentz-scalar traces.
!Result res(nout, diagram 1..3, channel 1 s / 2 a / 3 m), un-normalised (divide by the normalisation B).
module current1_solve
    use libraries
    use quark_elems
    use diquark_elems
    use faddeev_elems
    use faddeev_solve
    use probe_elems
    use current1_elems
    implicit none

private

type current1_equation
    type(current_para),public:: para
    type(current_grid),public:: gr
contains
    procedure,public:: init     => current1_equation_init
    procedure,public:: diagrams => current1_equation_diagrams
end type current1_equation

public current1_equation

contains

    subroutine current1_equation_init(this,para)
        class(current1_equation):: this
        type(current_para):: para
        this%para = para
        this%gr = make_grid(para%nk,para%nz,para%ny,0,para%scale)
    end subroutine

    subroutine current1_equation_diagrams(this,fe,pr,Q,charge,phi,res)
        class(current1_equation):: this
        type(faddeev_equation):: fe
        type(probe_type):: pr
        real(8):: Q,charge(2,2),phi(2)
        complex(8),allocatable:: res(:,:,:)
        complex(8),allocatable:: Tt(:,:,:,:),Map(:,:),s(:,:,:),vq(:,:,:),c0(:),c1(:,:,:),t01(:,:)
        complex(8),allocatable:: vsa(:,:),vas(:,:),J(:,:,:)
        complex(8):: Pf(4),Pi(4),Qv(4),k(4),p(4),kq(4),kd(4),pq(4),pd(4)
        complex(8):: Ps(4,4),Pa(4,4,4),Cs(4,4),Ca(4,4,4),Ya(4,4,4),SPs(4,4),SPa(4,4,4),Sk(4,4)
        complex(8):: D1k(4,4),D1p(4,4),D0k,D0p
        real(8):: f1(2),f2(2),f3(2,2),eta,etb,w
        integer:: ip,m,al,be,nb,nidx,d,ch,nout
        eta = this%para%eta; etb = 1-eta
        call kinematics(fe%M,Q,Pf,Pi,Qv)
        call pr%projectors(Pf,Pi,Qv,Tt,Map)
        nb = size(Tt,4); nidx = pr%nidx; nout = size(Map,1)
        call flavour_factors(pr,charge,phi,f1,f2,f3)
        allocate(s(nb,3,3),J(4,4,nidx),vsa(4,nidx),vas(4,nidx)); s = Zero
        do ip=1,this%gr%n
            if(.not.MPI_distribute(ip)) cycle
            k = this%gr%k(:,ip); w = this%gr%w(ip)
            call fe%amplitude(k,Pi,Ps,Pa)
            kq = eta*Pi+k; kd = etb*Pi-k
            D0k = Dprop0(dot4(kd,kd)); D1k = Dprop1mat(kd)
            !---- diagram 1: probe on the bystander quark
            p = k+etb*Qv
            call fe%conj_amplitude(p,Pf,Cs,Ca)
            pq = eta*Pf+p
            vq = pr%quark(pq,kq)
            do al=1,4
                Ya(:,:,al) = Zero
                do be=1,4
                    Ya(:,:,al) = Ya(:,:,al)+D1k(al,be)*Pa(:,:,be)
                end do
            end do
            do m=1,nidx
                J(:,:,m) = f1(1)*D0k*matmul(Cs,matmul(vq(:,:,m),Ps))
            end do
            call accumulate(s(:,1,1))
            do m=1,nidx
                J(:,:,m) = Zero
                do al=1,4
                    J(:,:,m) = J(:,:,m)+f1(2)*matmul(Ca(:,:,al),matmul(vq(:,:,m),Ya(:,:,al)))
                end do
            end do
            call accumulate(s(:,1,2))
            !---- diagrams 2, 3: probe on the diquark
            p = k-eta*Qv
            call fe%conj_amplitude(p,Pf,Cs,Ca)
            pd = etb*Pf-p
            Sk = propagator(kq)
            SPs = matmul(Sk,Ps)
            do be=1,4
                SPa(:,:,be) = matmul(Sk,Pa(:,:,be))
            end do
            c0 = pr%dq_s(pd,kd)
            do m=1,nidx
                J(:,:,m) = f2(1)*c0(m)*matmul(Cs,SPs)
            end do
            call accumulate(s(:,2,1))
            c1 = pr%dq_a(pd,kd)
            do m=1,nidx
                J(:,:,m) = Zero
                do al=1,4; do be=1,4
                    J(:,:,m) = J(:,:,m)+f2(2)*c1(al,be,m)*matmul(Ca(:,:,al),SPa(:,:,be))
                end do; end do
            end do
            call accumulate(s(:,2,2))
            t01 = pr%trans(pd,kd)
            D0p = Dprop0(dot4(pd,pd)); D1p = Dprop1mat(pd)
            do m=1,nidx
                do be=1,4
                    vsa(be,m) = sum(t01(:,m)*D1k(:,be))*D0p
                end do
                do al=1,4
                    vas(al,m) = sum(D1p(al,:)*t01(:,m))*D0k
                end do
                J(:,:,m) = Zero
                do be=1,4
                    J(:,:,m) = J(:,:,m)+f3(1,2)*vsa(be,m)*matmul(Cs,SPa(:,:,be))
                end do
                do al=1,4
                    J(:,:,m) = J(:,:,m)+f3(2,1)*vas(al,m)*matmul(Ca(:,:,al),SPs)
                end do
            end do
            call accumulate(s(:,3,3))
        end do
        call MPI_reducesum(s(:,1,1)); call MPI_reducesum(s(:,1,2)); call MPI_reducesum(s(:,2,1))
        call MPI_reducesum(s(:,2,2)); call MPI_reducesum(s(:,3,3))
        if(allocated(res)) deallocate(res)
        allocate(res(nout,3,3)); res = Zero
        do d=1,3; do ch=1,3
            res(:,d,ch) = matmul(Map,s(:,d,ch))
        end do; end do
    contains
        subroutine accumulate(sv)
            complex(8):: sv(:)
            integer:: b,mm
            do b=1,nb
                do mm=1,nidx
                    sv(b) = sv(b)+w*tr2(Tt(:,:,mm,b),J(:,:,mm))
                end do
            end do
        end subroutine
    end subroutine

end module

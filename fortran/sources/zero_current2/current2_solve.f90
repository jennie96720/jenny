!Two-loop diagrams 4-6 of the nucleon current, from gauging OUR Faddeev kernel (= qdq/exchange.py)
!
!  J^{4-6}_m = int_k int_l  Psibar_X(k;-P_f) G0_X(k) K^m_XY(k,l) G0_Y(l) Psi_Y(l;P_i)
!  K_XY = -4 g_X g_Y c_XY Gamma^Y(rel_in) S^T(q) Gammabar^X(rel_out),  q = l_qq - k_q, q' = k_qq - l_q
!  piece 1 (diagram 4): S^T(q) -> [probe quark vertex(q',q)]^T                (exchanged quark)
!  pieces 2,3 (diagram 5): Gamma^Y -> X^Y (outgoing legs: bystander k_q, exchanged q'), then S^T(q')
!  pieces 4,5 (diagram 6): Gammabar^X -> -Xbar^X (incoming legs: bystander l_q, exchanged q)
!  (graviton: legs weighted by their momenta, plus delta_{mu nu}/2 x amplitude; see probe_lift)
!
!Trace Tr[Tt out Gamma^Y_rho mid Gammabar^X_mu' in_rho] is evaluated as Tr[mid W] with
!W = Bp Ap, Bp = Gammabar^X_mu' in_rho (inner only), Ap = Tt out_mu' Gamma^Y_rho (outer only): one zgemm
!per outer point and block.
module current2_solve
    use libraries
    use quark_elems
    use diquark_elems
    use faddeev_elems
    use faddeev_solve
    use probe_elems
    use current1_elems
    implicit none

private

type current2_equation
    type(current_para),public:: para
    type(current_grid),public:: go,gi
contains
    procedure,public:: init     => current2_equation_init
    procedure,public:: compute  => current2_equation_compute
end type current2_equation

public current2_equation

contains

    subroutine current2_equation_init(this,para)
        class(current2_equation):: this
        type(current_para):: para
        this%para = para
        this%go = make_grid(para%nko(1),para%nko(2),para%nko(3),0,para%scale)
        this%gi = make_grid(para%nli(1),para%nli(2),para%nli(3),para%nli(4),para%scale)
    end subroutine

    !res(nout, piece 1..5, block XY = 1 ss, 2 sa, 3 as, 4 aa), flavour stripped, un-normalised.
    !plain = .true.: Psibar G0 K G0 Psi at Q = 0 (no probe; returns its trace/2 in res(1,1,:)).
    subroutine current2_equation_compute(this,fe,pr,Q,res,plain)
        class(current2_equation):: this
        type(faddeev_equation):: fe
        type(probe_type):: pr
        real(8):: Q
        complex(8),allocatable:: res(:,:,:)
        logical,optional:: plain
        logical:: pl
        complex(8),allocatable:: Tt(:,:,:,:),Map(:,:),outs(:,:,:),outa(:,:,:,:),inns(:,:,:),inna(:,:,:,:)
        complex(8),allocatable:: BpM(:,:,:),ApM(:,:),W(:,:),mid(:,:,:,:,:),acc(:,:,:),chiT(:,:,:),vq(:,:,:)
        complex(8),allocatable:: lq(:,:),lqq(:,:)
        complex(8):: Pf(4),Pi(4),Qv(4),k(4),kq(4),kqq(4),qe(4),qp(4),rin(4),rout(4),l5(4),l6(4)
        complex(8):: Ps(4,4),Pa(4,4,4),Cs(4,4),Ca(4,4,4),D1(4,4),Sk(4,4),SqT(4,4),SqpT(4,4),vec(4,4,4),smid(4,4)
        complex(8):: Fin(2),Fout(2),F5(2),F6(2),d5m(2),d5p(2),d6p(2),d6m(2),pref,g(2),gamY(4,4,4),barX(4,4,4)
        integer:: nb,nidx,no,ni,io,n,X,Y,blk,nmp,nrh,m,b,a,z,mu,rho,al,nu,pc,ncol,nrow,kdim
        pl = .false.; if(present(plain)) pl = plain
        g = [fe%gs,fe%ga]
        if(pl)then
            Pf = [Zero,Zero,Zero,Im*fe%M]; Pi = Pf; Qv = Zero
            nb = 1; nidx = 1
            allocate(Tt(4,4,1,1),Map(1,1)); Tt(:,:,1,1) = id4; Map = One
        else
            call kinematics(fe%M,Q,Pf,Pi,Qv)
            call pr%projectors(Pf,Pi,Qv,Tt,Map)
            nb = size(Tt,4); nidx = pr%nidx
        end if
        no = this%go%n; ni = this%gi%n
        !outer: Psibar G0 (k);  inner: G0 Psi (l)
        allocate(outs(4,4,no),outa(4,4,4,no),inns(4,4,ni),inna(4,4,4,ni),lq(4,ni),lqq(4,ni))
        do io=1,no
            k = this%go%k(:,io)
            call fe%conj_amplitude(k,Pf,Cs,Ca)
            kq = this%para%eta*Pf+k; kqq = (1-this%para%eta)*Pf-k
            Sk = propagator(kq); D1 = Dprop1mat(kqq)
            outs(:,:,io) = Dprop0(dot4(kqq,kqq))*matmul(Cs,Sk)
            do mu=1,4
                outa(:,:,mu,io) = Zero
                do al=1,4
                    outa(:,:,mu,io) = outa(:,:,mu,io)+D1(al,mu)*matmul(Ca(:,:,al),Sk)
                end do
            end do
        end do
        do n=1,ni
            call fe%amplitude(this%gi%k(:,n),Pi,Ps,Pa)
            lq(:,n) = this%para%eta*Pi+this%gi%k(:,n); lqq(:,n) = (1-this%para%eta)*Pi-this%gi%k(:,n)
            Sk = propagator(lq(:,n)); D1 = Dprop1mat(lqq(:,n))
            inns(:,:,n) = Dprop0(dot4(lqq(:,n),lqq(:,n)))*matmul(Sk,Ps)
            do rho=1,4
                inna(:,:,rho,n) = Zero
                do nu=1,4
                    inna(:,:,rho,n) = inna(:,:,rho,n)+D1(rho,nu)*matmul(Sk,Pa(:,:,nu))
                end do
            end do
        end do
        allocate(acc(nb,5,4),mid(4,4,nidx,ni,5),chiT(4,4,nidx)); acc = Zero
        nrow = 4*ni; ncol = 4*nb*nidx
        allocate(BpM(4,ni,64),ApM(64,ncol),W(nrow,ncol))          !allocated once, reused
        do io=1,no
            if(.not.MPI_distribute(io)) cycle
            k = this%go%k(:,io)
            kq = this%para%eta*Pf+k; kqq = (1-this%para%eta)*Pf-k
            do blk=1,4
                X = merge(1,2,blk<=2); Y = merge(1,2,blk==1.or.blk==3)
                nmp = merge(1,4,X==1); nrh = merge(1,4,Y==1)
                if(Y==1)then
                    gamY(:,:,1) = gam_s
                else
                    gamY = gam_a
                end if
                if(X==1)then
                    barX(:,:,1) = bar_s
                else
                    barX = bar_a
                end if
                !mid matrices for this outer point and block
                do n=1,ni
                    qe = lqq(:,n)-kq; qp = kqq-lq(:,n)
                    rin = (kq-qe)/2; rout = (lq(:,n)-qp)/2
                    l5 = (kq-qp)/2; l6 = (lq(:,n)-qe)/2
                    SqT = transpose(propagator(qe))
                    Fin(Y) = Famp(tc(Y),dot4(rin,rin)); Fout(X) = Famp(tc(X),dot4(rout,rout))
                    if(pl)then
                        mid(:,:,1,n,1) = Fin(Y)*Fout(X)*SqT
                        cycle
                    end if
                    SqpT = transpose(propagator(qp))
                    vq = pr%quark(qp,qe)
                    do m=1,nidx
                        chiT(:,:,m) = transpose(vq(:,:,m))
                    end do
                    F5(Y) = Famp(tc(Y),dot4(l5,l5)); F6(X) = Famp(tc(X),dot4(l6,l6))
                    d5m(Y) = divdiff(4+Y,dot4(l5-Qv/2,l5-Qv/2),dot4(l5,l5))
                    d5p(Y) = divdiff(4+Y,dot4(l5+Qv/2,l5+Qv/2),dot4(l5,l5))
                    d6p(X) = divdiff(4+X,dot4(l6+Qv/2,l6+Qv/2),dot4(l6,l6))
                    d6m(X) = divdiff(4+X,dot4(l6-Qv/2,l6-Qv/2),dot4(l6,l6))
                    mid(:,:,:,n,1) = Fin(Y)*Fout(X)*chiT
                    smid = F5(Y)*Fout(X)*SqpT
                    do mu=1,4
                        vec(:,:,mu) = -0.25d0*(4*l5(mu)-Qv(mu))*d5m(Y)*Fout(X)*SqpT
                    end do
                    mid(:,:,:,n,2) = pr%lift(vec,kq,smid)
                    do mu=1,4
                        vec(:,:,mu) = 0.25d0*(4*l5(mu)+Qv(mu))*d5p(Y)*Fout(X)*SqpT
                    end do
                    mid(:,:,:,n,3) = pr%lift(vec,qp,smid)
                    smid = Fin(Y)*F6(X)*SqT
                    do mu=1,4
                        vec(:,:,mu) = -0.25d0*(4*l6(mu)+Qv(mu))*d6p(X)*Fin(Y)*SqT
                    end do
                    mid(:,:,:,n,4) = pr%lift(vec,lq(:,n),smid)
                    do mu=1,4
                        vec(:,:,mu) = 0.25d0*(4*l6(mu)-Qv(mu))*d6m(X)*Fin(Y)*SqT
                    end do
                    mid(:,:,:,n,5) = pr%lift(vec,qe,smid)
                end do
                !W = Bp Ap :  rows (a, n), inner index (x, mu', rho), columns (z, b, m)
                kdim = 4*nmp*nrh
                do n=1,ni
                    do mu=1,nmp; do rho=1,nrh
                        Sk = matmul(barX(:,:,mu),merge(inns(:,:,n),inna(:,:,rho,n),Y==1))
                        do z=1,4
                            BpM(:,n,z+4*(mu-1)+4*nmp*(rho-1)) = Sk(:,z)
                        end do
                    end do; end do
                end do
                do m=1,nidx; do b=1,nb
                    do mu=1,nmp; do rho=1,nrh
                        Sk = matmul(Tt(:,:,m,b),matmul(merge(outs(:,:,io),outa(:,:,mu,io),X==1),gamY(:,:,rho)))
                        do a=1,4
                            ApM(a+4*(mu-1)+4*nmp*(rho-1),1+4*(b-1)+4*nb*(m-1):4+4*(b-1)+4*nb*(m-1)) = Sk(a,:)
                        end do
                    end do; end do
                end do; end do
                call zgemm('N','N',nrow,ncol,kdim,One,BpM,nrow,ApM,64,Zero,W,nrow)
                pref = -4*g(X)*g(Y)*this%go%w(io)
                do pc=1,merge(1,5,pl)
                    do b=1,nb
                        do n=1,ni
                            do m=1,nidx
                                acc(b,pc,blk) = acc(b,pc,blk)+pref*this%gi%w(n)* &
                                    sum(W(1+4*(n-1):4+4*(n-1),1+4*(b-1)+4*nb*(m-1):4+4*(b-1)+4*nb*(m-1))* &
                                        transpose(mid(:,:,m,n,pc)))
                            end do
                        end do
                    end do
                end do
            end do
        end do
        do blk=1,4
            do pc=1,5
                call MPI_reducesum(acc(:,pc,blk))
            end do
        end do
        if(allocated(res)) deallocate(res)
        allocate(res(size(Map,1),5,4))
        do blk=1,4; do pc=1,5
            if(pl)then
                res(:,pc,blk) = acc(:,pc,blk)/2
            else
                res(:,pc,blk) = matmul(Map,acc(:,pc,blk))
            end if
        end do; end do
    end subroutine

    pure character function tc(X)
        integer,intent(in):: X
        tc = merge('s','a',X==1)
    end function

end module

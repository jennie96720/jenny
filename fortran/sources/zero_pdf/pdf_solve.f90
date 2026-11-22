!Weighted forward currents for the DF moments (= qdq/pdf.py, qdq/pdf_pol.py, qdq/diquark_df.py)
!
!pdf_one_loop:  res(lam, m, comp), comp
!   number current (pol = .false.):  1 q_s, 2 q_a (probe on the bystander quark, weight k_q)
!                                    3 D_s, 4 D_a (probe on the diquark, weight k_d: diquark LF distribution)
!   axial current  (pol = .true.):   1 q_s, 2 q_a,  3 D_aa, 4 D_sa, 5 D_as (couplings at each kd^2)
!pdf_two_loop:  acc(lam, m, piece, block), flavour stripped
!   number current: pieces 1 ex4 (weight q), 2 by5 (k_q), 3 ex5 (q'), 4 by6 (l_q), 5 ex6 (q)
!   axial current:  piece 1 ex4 only (the seagulls ~ Q_mu/(Q^2 + m_pi^2) vanish at Q = 0)
!   same W = Bp Ap zgemm structure as current2_solve, with Tt = n_mu Y.
!da_moments:  <x^m> of the 0+ / 1+ diquark leading-twist DA (normalised by m = 0), per lam
module pdf_solve
    use libraries
    use quark_elems
    use diquark_elems
    use faddeev_elems
    use faddeev_solve
    use probe_elems
    use current1_elems
    use pdf_elems
    use axial_elems
    implicit none

contains

    subroutine pdf_one_loop(fe,gr,para,P,charge,phi,pol,tab,res)
        type(faddeev_equation):: fe
        type(current_grid):: gr
        type(pdf_para):: para
        complex(8):: P(4)
        real(8):: charge(2,2),phi(2)
        logical:: pol
        type(axial_table):: tab
        complex(8),allocatable:: res(:,:,:)
        type(probe_type):: em
        complex(8),allocatable:: v3(:,:,:),c0(:),c1(:,:,:)
        complex(8):: k(4),kq(4),kd(4),Ps(4,4),Pa(4,4,4),Cs(4,4),Ca(4,4,4),D1(4,4),D0,vq(4,4),Ya(4,4,4),Sk(4,4)
        complex(8):: SPs(4,4),SPa(4,4,4),T(4,4),TV(4,4),Xp(4,4),tc(5),nP,kd2,cpl(5),Gaa(4,4),cn(4,4),vsa(4),vas(4)
        complex(8):: nD1(4),D1n(4),nkd
        complex(8),allocatable:: wq(:,:),wd(:,:)
        real(8):: f1(2),f2(2),f3(2,2),w,eta
        integer:: ip,mu,al,be,a,b,ncomp,nl,mm,c
        nl = para%nlam; mm = para%mmax; eta = para%eta
        em = probe_create('em',fe%M)
        call flavour_factors(em,charge,phi,f1,f2,f3)
        call lf_structures(P,fe%M,T,TV,Xp)
        if(.not.pol) Xp = id4
        ncomp = merge(5,4,pol)
        if(allocated(res)) deallocate(res)
        allocate(res(nl,0:mm,ncomp),wq(nl,0:mm),wd(nl,0:mm)); res = Zero
        nP = dot4(nvec,P)
        do ip=1,gr%n
            if(.not.MPI_distribute(ip)) cycle
            k = gr%k(:,ip); w = gr%w(ip)
            call fe%amplitude(k,P,Ps,Pa)
            call fe%conj_amplitude(k,P,Cs,Ca)
            kq = eta*P+k; kd = (1-eta)*P-k
            D0 = Dprop0(dot4(kd,kd)); D1 = Dprop1mat(kd)
            !---- diagram 1: probe on the bystander quark
            if(pol)then
                v3 = axial_quark(kq)
            else
                v3 = em%quark(kq,kq)
            end if
            vq = Zero
            do mu=1,4
                vq = vq+nvec(mu)*v3(:,:,mu)
            end do
            do al=1,4
                Ya(:,:,al) = Zero
                do be=1,4
                    Ya(:,:,al) = Ya(:,:,al)+D1(al,be)*Pa(:,:,be)
                end do
            end do
            tc = Zero
            tc(1) = f1(1)*D0*tr2(Xp,matmul(Cs,matmul(vq,Ps)))
            do al=1,4
                tc(2) = tc(2)+f1(2)*tr2(Xp,matmul(Ca(:,:,al),matmul(vq,Ya(:,:,al))))
            end do
            !---- diagrams 2, 3: probe on the diquark
            Sk = propagator(kq)
            SPs = matmul(Sk,Ps)
            do be=1,4
                SPa(:,:,be) = matmul(Sk,Pa(:,:,be))
            end do
            if(.not.pol)then
                c0 = em%dq_s(kd,kd)
                tc(3) = f2(1)*sum(nvec*c0)*tr2(Xp,matmul(Cs,SPs))
                c1 = em%dq_a(kd,kd)
                do al=1,4; do be=1,4
                    tc(4) = tc(4)+f2(2)*sum(nvec*c1(al,be,:))*tr2(Xp,matmul(Ca(:,:,al),SPa(:,:,be)))
                end do; end do
            else
                kd2 = dot4(kd,kd)
                cpl = tab%eval(kd2)
                do a=1,4; do b=1,4
                    Gaa(a,b) = Zero
                    do mu=1,4
                        Gaa(a,b) = Gaa(a,b)+cpl(1)*nvec(mu)*sum(eps4(mu,a,b,:)*kd)
                    end do
                end do; end do
                cn = matmul(D1,matmul(Gaa,D1))
                do al=1,4; do be=1,4
                    tc(3) = tc(3)+f2(2)*cn(al,be)*tr2(Xp,matmul(Ca(:,:,al),SPa(:,:,be)))
                end do; end do
                nkd = dot4(kd,nvec)/kd2
                vsa = cpl(2)*nvec+cpl(3)*nkd*kd
                vas = cpl(4)*nvec+cpl(5)*nkd*kd
                do b=1,4
                    nD1(b) = sum(vsa*D1(:,b))
                    D1n(b) = sum(D1(b,:)*vas)
                end do
                do b=1,4
                    tc(4) = tc(4)+f3(1,2)*D0*nD1(b)*tr2(Xp,matmul(Cs,SPa(:,:,b)))
                    tc(5) = tc(5)+f3(2,1)*D0*D1n(b)*tr2(Xp,matmul(Ca(:,:,b),SPs))
                end do
            end if
            wq = lf_weights(dot4(kq,nvec)/nP,dot4(kq,kq),para%lams,nl,mm)
            wd = lf_weights(dot4(kd,nvec)/nP,dot4(kd,kd),para%lams,nl,mm)
            res(:,:,1) = res(:,:,1)+w*tc(1)*wq
            res(:,:,2) = res(:,:,2)+w*tc(2)*wq
            do c=3,ncomp
                res(:,:,c) = res(:,:,c)+w*tc(c)*wd
            end do
        end do
        do c=1,ncomp
            call MPI_reducesum(res(:,:,c))
        end do
    end subroutine

    subroutine pdf_two_loop(fe,go,gi,para,P,pol,acc)
        type(faddeev_equation):: fe
        type(current_grid):: go,gi
        type(pdf_para):: para
        complex(8):: P(4)
        logical:: pol
        complex(8),allocatable:: acc(:,:,:,:)
        type(probe_type):: em
        complex(8),allocatable:: outs(:,:,:),outa(:,:,:,:),inns(:,:,:),inna(:,:,:,:),BpM(:,:,:),ApM(:,:),W(:,:)
        complex(8),allocatable:: mid(:,:,:,:,:),lq(:,:),lqq(:,:),tsum(:,:,:),wt(:,:),v3(:,:,:)
        complex(8):: k(4),kq(4),kqq(4),qe(4),qp(4),rin(4),rout(4),l5(4),l6(4),Ps(4,4),Pa(4,4,4),Cs(4,4),Ca(4,4,4)
        complex(8):: D1(4,4),Sk(4,4),SqT(4,4),SqpT(4,4),Fin(2),Fout(2),d5(2),d6(2),pref(4),g(2),gamY(4,4,4),barX(4,4,4)
        complex(8):: Tt(4,4,4),T(4,4),TV(4,4),Xp(4,4),nP,mom(4,5)
        integer:: no,ni,io,n,X,Y,blk,nmp,nrh,m,a,z,mu,rho,al,nu,pc,npc,ncol,nrow,kdim,nl,mm
        nl = para%nlam; mm = para%mmax
        npc = merge(1,5,pol)
        g = [fe%gs,fe%ga]
        em = probe_create('em',fe%M)
        call lf_structures(P,fe%M,T,TV,Xp)
        if(.not.pol) Xp = id4
        do mu=1,4
            Tt(:,:,mu) = nvec(mu)*Xp
        end do
        nP = dot4(nvec,P)
        no = go%n; ni = gi%n
        allocate(outs(4,4,no),outa(4,4,4,no),inns(4,4,ni),inna(4,4,4,ni),lq(4,ni),lqq(4,ni))
        do io=1,no
            k = go%k(:,io)
            call fe%conj_amplitude(k,P,Cs,Ca)
            kq = para%eta*P+k; kqq = (1-para%eta)*P-k
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
            call fe%amplitude(gi%k(:,n),P,Ps,Pa)
            lq(:,n) = para%eta*P+gi%k(:,n); lqq(:,n) = (1-para%eta)*P-gi%k(:,n)
            Sk = propagator(lq(:,n)); D1 = Dprop1mat(lqq(:,n))
            inns(:,:,n) = Dprop0(dot4(lqq(:,n),lqq(:,n)))*matmul(Sk,Ps)
            do rho=1,4
                inna(:,:,rho,n) = Zero
                do nu=1,4
                    inna(:,:,rho,n) = inna(:,:,rho,n)+D1(rho,nu)*matmul(Sk,Pa(:,:,nu))
                end do
            end do
        end do
        if(allocated(acc)) deallocate(acc)
        allocate(acc(nl,0:mm,npc,4),mid(4,4,4,ni,npc),tsum(ni,npc,4),wt(nl,0:mm)); acc = Zero
        nrow = 4*ni; ncol = 16
        allocate(BpM(4,ni,64),ApM(64,ncol),W(nrow,ncol))
        do io=1,no
            if(.not.MPI_distribute(io)) cycle
            k = go%k(:,io)
            kq = para%eta*P+k; kqq = (1-para%eta)*P-k
            tsum = Zero
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
                do n=1,ni
                    qe = lqq(:,n)-kq; qp = kqq-lq(:,n)                  !Q = 0: q' = q
                    rin = (kq-qe)/2; rout = (lq(:,n)-qp)/2
                    l5 = (kq-qp)/2; l6 = (lq(:,n)-qe)/2
                    Fin(Y) = Famp(merge('s','a',Y==1),dot4(rin,rin)); Fout(X) = Famp(merge('s','a',X==1),dot4(rout,rout))
                    if(pol)then
                        v3 = axial_quark(qe)
                    else
                        v3 = em%quark(qp,qe)
                    end if
                    do mu=1,4
                        mid(:,:,mu,n,1) = Fin(Y)*Fout(X)*transpose(v3(:,:,mu))
                    end do
                    if(pol) cycle
                    SqT = transpose(propagator(qe)); SqpT = transpose(propagator(qp))
                    d5(Y) = divdiff(4+Y,dot4(l5,l5),dot4(l5,l5))
                    d6(X) = divdiff(4+X,dot4(l6,l6),dot4(l6,l6))
                    do mu=1,4
                        mid(:,:,mu,n,2) = -l5(mu)*d5(Y)*Fout(X)*SqpT
                        mid(:,:,mu,n,3) =  l5(mu)*d5(Y)*Fout(X)*SqpT
                        mid(:,:,mu,n,4) = -l6(mu)*d6(X)*Fin(Y)*SqT
                        mid(:,:,mu,n,5) =  l6(mu)*d6(X)*Fin(Y)*SqT
                    end do
                end do
                kdim = 4*nmp*nrh
                do n=1,ni
                    do mu=1,nmp; do rho=1,nrh
                        Sk = matmul(barX(:,:,mu),merge(inns(:,:,n),inna(:,:,rho,n),Y==1))
                        do z=1,4
                            BpM(:,n,z+4*(mu-1)+4*nmp*(rho-1)) = Sk(:,z)
                        end do
                    end do; end do
                end do
                do m=1,4
                    do mu=1,nmp; do rho=1,nrh
                        Sk = matmul(Tt(:,:,m),matmul(merge(outs(:,:,io),outa(:,:,mu,io),X==1),gamY(:,:,rho)))
                        do a=1,4
                            ApM(a+4*(mu-1)+4*nmp*(rho-1),1+4*(m-1):4+4*(m-1)) = Sk(a,:)
                        end do
                    end do; end do
                end do
                call zgemm('N','N',nrow,ncol,kdim,One,BpM,nrow,ApM,64,Zero,W,nrow)
                pref(blk) = -4*g(X)*g(Y)*go%w(io)
                do pc=1,npc
                    do n=1,ni
                        do m=1,4
                            tsum(n,pc,blk) = tsum(n,pc,blk)+sum(W(1+4*(n-1):4+4*(n-1),1+4*(m-1):4+4*(m-1))* &
                                transpose(mid(:,:,m,n,pc)))
                        end do
                    end do
                end do
            end do
            !weights of the struck line in each piece: ex4 q, by5 k_q, ex5 q', by6 l_q, ex6 q
            do n=1,ni
                qe = lqq(:,n)-kq; qp = kqq-lq(:,n)
                mom(:,1) = qe; mom(:,2) = kq; mom(:,3) = qp; mom(:,4) = lq(:,n); mom(:,5) = qe
                do pc=1,npc
                    wt = lf_weights(dot4(mom(:,pc),nvec)/nP,dot4(mom(:,pc),mom(:,pc)),para%lams,nl,mm)
                    do blk=1,4
                        acc(:,:,pc,blk) = acc(:,:,pc,blk)+pref(blk)*gi%w(n)*tsum(n,pc,blk)*wt
                    end do
                end do
            end do
        end do
        do blk=1,4; do pc=1,npc
            call MPI_reducesum(acc(:,:,pc,blk))
        end do; end do
    end subroutine

    !<x^m> (per lam, normalised by m = 0) of the 0+ (ch = 1) or 1+ (ch = 2) diquark DA:
    !  0+: Tr[gamma_5 gamma.n S(k_+) C i gamma_5 C^dag S(k_-)] F,  1+: Tr[gamma.n S(k_+) i gamma.n S(k_-)] F
    subroutine da_moments(ch,para,mom)
        integer:: ch
        type(pdf_para):: para
        complex(8),allocatable:: mom(:,:)
        type(current_grid):: gr
        complex(8):: P(4),kp(4),km(4),GC(4,4),proj(4,4),tval,nP,k(4)
        complex(8),allocatable:: wt(:,:)
        real(8):: m
        integer:: ip,nl,mm,l
        nl = para%nlam; mm = para%mmax
        m = merge(dq_m0,dq_m1,ch==1)
        P = [Zero,Zero,Zero,Im*m]
        gr = make_grid(para%da(1),para%da(2),para%da(3),0,1d0)
        call compact_grid(gr,para%da_kmax)
        if(ch==1)then
            GC = matmul(cc,matmul(Im*g5,ccdag)); proj = matmul(g5,slash(nvec))
        else
            GC = Im*slash(nvec); proj = slash(nvec)
        end if
        if(allocated(mom)) deallocate(mom)
        allocate(mom(nl,0:mm),wt(nl,0:mm)); mom = Zero
        nP = dot4(nvec,P)
        do ip=1,gr%n
            if(.not.MPI_distribute(ip)) cycle
            k = gr%k(:,ip)
            kp = k+P/2; km = k-P/2
            tval = tr2(proj,matmul(propagator(kp),matmul(GC,propagator(km))))*calF(dot4(k,k)/(0.5d0*m**2))
            wt = lf_weights(dot4(kp,nvec)/nP,dot4(kp,kp),para%lams,nl,mm)
            mom = mom+gr%w(ip)*tval*wt
        end do
        call MPI_reducesum(mom)
        do l=1,nl
            mom(l,:) = mom(l,:)/mom(l,0)
        end do
    end subroutine

end module

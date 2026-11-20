!Canonical normalisation from the Faddeev eigenvalue (Ward identity), = qdq/current.py normalisation:
!   full baryon charge  J^B_mu(0) = -dlambda/dP_mu  int Psibar G0 Psi = (P_mu/M)(dlambda/dM) n0 Lambda_+
!Also returns -Psibar dG0/dP Psi (projected on T1) = diagrams 1+2 with baryon charges at Q = 0 (a check).
module normalization_solve
    use libraries
    use quark_elems
    use diquark_elems
    use faddeev_elems
    use faddeev_solve
    use probe_elems
    use current1_elems
    implicit none

contains

    subroutine normalisation(fe,gr,eta,B,F1G0,dlam,n0)
        type(faddeev_equation):: fe
        type(current_grid):: gr
        real(8):: eta
        complex(8):: B,F1G0,dlam,n0
        type(probe_type):: em
        complex(8),allocatable:: Tt(:,:,:,:),Map(:,:),vec(:)
        complex(8):: P(4),Ps(4,4),Pa(4,4,4),Cs(4,4),Ca(4,4,4),e(4),dJ(4,4,4),JB(4,4,4),lp,lm,acc(1),G0(4,4)
        real(8),parameter:: h = 1d-4, dM = 1d-3
        integer:: ip,mu
        P = [Zero,Zero,Zero,Im*fe%M]
        em = probe_create('em',fe%M)
        call em%projectors(P,P,[Zero,Zero,Zero,Zero],Tt,Map)
        call fe%eigen(fe%M+dM,lp,vec)
        call fe%eigen(fe%M-dM,lm,vec)
        dlam = (lp-lm)/(2*dM)
        n0 = Zero; dJ = Zero
        do ip=1,gr%n
            if(.not.MPI_distribute(ip)) cycle
            call fe%amplitude(gr%k(:,ip),P,Ps,Pa)
            call fe%conj_amplitude(gr%k(:,ip),P,Cs,Ca)
            G0 = G0sand(P,gr%k(:,ip))
            n0 = n0+gr%w(ip)*(G0(1,1)+G0(2,2)+G0(3,3)+G0(4,4))/2
            do mu=1,4
                e = Zero; e(mu) = h
                dJ(:,:,mu) = dJ(:,:,mu)-gr%w(ip)*(G0sand(P+e,gr%k(:,ip))-G0sand(P-e,gr%k(:,ip)))/(2*h)
            end do
        end do
        acc(1) = n0; call MPI_reducesum(acc); n0 = acc(1)
        do mu=1,4
            call MPI_reducesum(dJ(:,:,mu))
        end do
        F1G0 = Zero; B = Zero
        do mu=1,4
            F1G0 = F1G0+Map(1,1)*tr2(Tt(:,:,mu,1),dJ(:,:,mu))
            JB(:,:,mu) = P(mu)/fe%M*dlam*n0*lam_plus(P,fe%M)
            B = B+Map(1,1)*tr2(Tt(:,:,mu,1),JB(:,:,mu))
        end do
    contains
        function G0sand(Pt,k) result(js)          !Psibar G0(k;Pt) Psi, Psi's held fixed
            complex(8):: Pt(4),k(4),js(4,4),kq(4),kd(4),Sq(4,4),D1(4,4)
            integer:: al,be
            kq = eta*Pt+k; kd = (1-eta)*Pt-k
            Sq = propagator(kq)
            js = matmul(Cs,matmul(Sq,Ps))*Dprop0(dot4(kd,kd))
            D1 = Dprop1mat(kd)
            do al=1,4; do be=1,4
                js = js+D1(al,be)*matmul(Ca(:,:,al),matmul(Sq,Pa(:,:,be)))
            end do; end do
        end function
    end subroutine

end module

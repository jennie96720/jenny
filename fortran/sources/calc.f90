!Drivers.  run_checks reproduces docs/reference_values.md stage by stage.
subroutine run_checks(fstr)
    use libraries
    use quark_elems
    use quark_utils
    use diquark_elems
    use diquark_solve
    use diquark_utils
    use faddeev_elems
    use faddeev_utils
    use faddeev_solve
    implicit none
    character(len=*):: fstr
    type(diquark_equation):: dq
    type(faddeev_equation):: fe
    type(faddeev_para):: fp
    real(8):: err,a,b,cm,c(2,2)
    complex(8):: sv,ss,lam,p2(5)
    complex(8),allocatable:: vec(:)
    integer:: i,ig,ic
    integer:: grids(3,2)
    character(len=2):: chs(2)

    call quark_para_load(fstr)
    if(jrank==0)then
        call dirac_check(err)
        print '(a,es10.2)','[1] Dirac algebra / charge conjugation max error: ',err
        p2 = [(0d0,0d0),(0.1d0,0d0),(1d0,0d0),(-0.1d0,0.2d0),(-0.25d0,0.5d0)]
        print '(a)','    p^2                  sigma_V                      sigma_S                      M(p^2)'
        do i=1,5
            call sigmas(p2(i),sv,ss)
            print '(2f7.3,3(2x,2f13.9))',p2(i),sv,ss,mass_function(p2(i))
        end do
        a = 0.05d0; b = 2d0
        do i=1,100
            cm = 0.5d0*(a+b)
            if(fme(a)*fme(cm)<=0)then
                b = cm
            else
                a = cm
            end if
        end do
        print '(a,f14.10,a)','    M^E = ',cm,' GeV   (reference 0.3275076942)'
    end if

    if(jrank==0) print '(a)','[2] diquark normalisation  (reference g^2: 56.868825, -42.400818)'
    call dq%init(diquark_para_load(fstr))
    call dq%solve()
    call dq%output()

    c = flavour_coeffs([1d0,0d0])
    if(jrank==0) print '(a,4f14.10)','[3] c_ss, c_sa, c_as, c_aa = ',c(1,1),c(1,2),c(2,1),c(2,2)

    if(jrank==0) print '(a)','[4] Faddeev eigenvalue on exact grids (Nystrom)  -- compare docs/reference_values.md'
    grids(:,1) = [8,4,4]; grids(:,2) = [16,8,8]
    chs = ['s ','sa']
    do ig=1,2
        do ic=1,2
            fp = faddeev_para_load(fstr)
            fp%nr = grids(1,ig); fp%nz = grids(2,ig); fp%ny = grids(3,ig); fp%ncheb = 0
            fp%channels = chs(ic); fp%own_g = .false.
            fp%gs = (7.47d0,0d0); fp%ga = (0d0,6.49d0)
            call fe%init(fp)
            call fe%eigen(1.10d0,lam,vec)
            if(jrank==0) print '(a,3i3,a,a3,a,f14.10)','    grid',grids(:,ig),'  channels ',chs(ic),'  lambda(1.10) =',real(lam)
            call fe%eigen(1.18d0,lam,vec)
            if(jrank==0) print '(a,3i3,a,a3,a,f14.10)','    grid',grids(:,ig),'  channels ',chs(ic),'  lambda(1.18) =',real(lam)
        end do
    end do

contains
    real(8) function fme(p)
        real(8):: p
        fme = p**2-real(mass_function(dcmplx(p**2,0d0)))**2
    end function
end subroutine

subroutine run_faddeev(fstr)
    use libraries
    use quark_utils
    use faddeev_utils
    use faddeev_solve
    implicit none
    character(len=*):: fstr
    type(faddeev_equation):: fe
    call quark_para_load(fstr)
    call diquark_init_from(fstr)
    call fe%init(faddeev_para_load(fstr))
    call fe%solve()
    call fe%output()
    if(jrank==0) print '(a,f10.5,a)','  M_N = ',fe%M,' GeV'
end subroutine

subroutine diquark_init_from(fstr)
    use diquark_elems
    use diquark_utils
    implicit none
    character(len=*):: fstr
    type(diquark_para):: dp
    dp = diquark_para_load(fstr)
    call diquark_elems_init(dp%m0,dp%m1)
end subroutine

!Form factors (proton; also neutron for 'em'): all six diagrams, normalised by the eigenvalue formula.
subroutine run_form_factors(fstr,kind)
    use libraries
    use quark_utils
    use faddeev_elems
    use faddeev_utils
    use faddeev_solve
    use probe_elems
    use current1_elems
    use current1_utils
    use current1_solve
    use current2_solve
    use normalization_solve
    implicit none
    character(len=*):: fstr,kind
    type(faddeev_equation):: fe
    type(current1_equation):: c1
    type(current2_equation):: c2
    type(current_para):: cp
    type(probe_type):: pr,em
    complex(8):: B,F1G0,dlam,n0,Kt
    complex(8),allocatable:: r1(:,:,:),r2(:,:,:),rB(:,:,:),d(:,:)
    real(8):: Q,eB(2,2),charge(2,2),phi(2,2),cx(2,2,5),ck(2,2),xq
    integer:: iq,inuc,blk,ix,iy,nout,j
    character(len=8):: names(5)
    call quark_para_load(fstr)
    call diquark_init_from(fstr)
    call fe%init(faddeev_para_load(fstr))
    call fe%solve()
    call fe%output()
    cp = current_para_load(fstr)
    call c1%init(cp)
    call c2%init(cp)
    call normalisation(fe,c1%gr,cp%eta,B,F1G0,dlam,n0)
    em = probe_create('em',fe%M)
    eB = 0d0; eB(1,1) = 1d0/3; eB(2,2) = 1d0/3
    call c1%diagrams(fe,em,0d0,eB,[1d0,0d0],rB)
    call c2%compute(fe,em,0d0,r2,plain=.true.)
    ck = flavour_coeffs([1d0,0d0])
    Kt = Zero
    do blk=1,4
        ix = merge(1,2,blk<=2); iy = merge(1,2,blk==1.or.blk==3)
        Kt = Kt+ck(ix,iy)*r2(1,1,blk)
    end do
    if(jrank==0)then
        print '(a,f10.5,a,f10.6)','  M_N = ',fe%M,' GeV,  dlambda/dM = ',real(dlam)
        print '(a,f12.7,a,f12.7)','  Psibar G0 K G0 Psi = ',real(Kt),'   Psibar G0 Psi = ',real(n0)
        print '(a,f12.7,a,f12.7,a)','  diagrams 1+2 (baryon) = ',aimag(sum(rB(1,1:2,1:2))),' i,  -Psibar dG0/dP Psi = ',aimag(F1G0),' i'
        print '(a,f12.7,a,f8.4)','  baryon charge (eigenvalue) = ',aimag(B),' i,  impulse share ',real(sum(rB(1,1:2,1:2))/B)
    end if
    pr = probe_create(kind,fe%M)
    if(kind=='grav')then
        names = ['A       ','J       ','D       ','S       ','cbar    ']
    else
        names = ['F1      ','F2      ','F3      ','        ','        ']
    end if
    phi = reshape([1d0,0d0,0d0,1d0],[2,2])
    if(jrank==0) call x_mkdir('outdata/'//trim(cp%root))
    if(jrank==0) open(unit=102,file='outdata/'//trim(cp%root)//'/'//trim(kind)//'_ff.txt',status='replace')
    do iq=1,cp%nQ2
        Q = sqrt(cp%Q2(iq)); xq = cp%Q2(iq)/fe%M**2
        call c2%compute(fe,pr,Q,r2)
        do inuc=1,merge(2,1,kind=='em')
            if(kind=='grav')then
                charge = reshape([1d0,0d0,0d0,1d0],[2,2])              !total EMT
            else
                charge = reshape([2d0/3,0d0,0d0,-1d0/3],[2,2])          !quark charges
            end if
            call c1%diagrams(fe,pr,Q,charge,phi(:,inuc),r1)
            cx = exchange_flavour(charge,phi(:,inuc))
            nout = size(r1,1)
            allocate(d(nout,6)); d = Zero
            d(:,1:3) = sum(r1,3)
            do blk=1,4
                ix = merge(1,2,blk<=2); iy = merge(1,2,blk==1.or.blk==3)
                d(:,4) = d(:,4)+cx(ix,iy,1)*r2(:,1,blk)
                d(:,5) = d(:,5)+cx(ix,iy,2)*r2(:,2,blk)+cx(ix,iy,3)*r2(:,3,blk)
                d(:,6) = d(:,6)+cx(ix,iy,4)*r2(:,4,blk)+cx(ix,iy,5)*r2(:,5,blk)
            end do
            d = d/B
            if(jrank==0)then
                print '(a,f8.4,a,f7.4,a,a1)','  Q2 =',cp%Q2(iq),' GeV^2 (x =',xq,')  nucleon ',merge('p','n',inuc==1)
                do j=1,nout
                    if(kind=='grav'.and.nout==2.and.j==2)then
                        print '(4x,a6,f10.5,a,6f9.4)',names(5),real(sum(d(j,:))),'   by diagram:',real(d(j,:))
                    else
                        print '(4x,a6,f10.5,a,6f9.4)',names(j),real(sum(d(j,:))),'   by diagram:',real(d(j,:))
                    end if
                end do
                if(kind=='em'.and.nout==3) print '(4x,a,2f10.5)','G_E, G_M =',real(sum(d(1,:))-xq/4*sum(d(2,:))), &
                    real(sum(d(1,:))+sum(d(2,:)))
                write(102,'(f10.5,i3,10es16.7)') cp%Q2(iq),inuc,(real(sum(d(j,:))),j=1,nout)
            end if
            deallocate(d)
        end do
    end do
    if(jrank==0) close(102)
end subroutine

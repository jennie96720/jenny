!Canonical normalisation of the 0+ and 1+ diquark amplitudes, 2611.19819 Eq. (A8):
!   2 K_mu = d/dQ_mu Tr int Gammabar(q;-K) S(q+Q/2) Gamma(q;K) S^T(-q+Q/2) |_{Q=K, K^2=-m^2}
!Contracting with K (and T_{ab} for 1+):  2K^2 [x3] = d/d alpha Pi(alpha K) at alpha = 1.
!Colour trace 2, unit-norm flavour matrices.  NB g1^2 < 0 in this gamma representation: g1 is imaginary.
module diquark_solve
    use libraries
    use quark_elems
    use diquark_elems
    use diquark_utils
    implicit none

private

type diquark_equation
    type(diquark_para),public:: para
    complex(8),public:: g2s,g2a,gs,ga
contains
    procedure,public:: init   => diquark_equation_init
    procedure,public:: solve  => diquark_equation_solve
    procedure,public:: output => diquark_equation_output
end type

public diquark_equation

contains

    subroutine diquark_equation_init(this,para)
        class(diquark_equation):: this
        type(diquark_para):: para
        this%para = para
        call diquark_elems_init(para%m0,para%m1)
    end subroutine

    !Pi(alpha K) (scalar) or T_{ab} Pi_{ab}(alpha K) (axial), coupling = 1, colour-flavour stripped
    function loop(this,ch,alpha) result(val)
        class(diquark_equation):: this
        character:: ch
        real(8):: alpha,m,q(this%para%nq),wq(this%para%nq),z(this%para%nz),wz(this%para%nz)
        complex(8):: val,K(4),Qv(4),qv4(4),Sp(4,4),SmT(4,4),F2,T(4,4),w
        integer:: i,j,a,b
        m = merge(this%para%m0,this%para%m1,ch=='s')
        K = [Zero,Zero,Zero,Im*m]
        Qv = alpha*K
        call radial(this%para%nq,1d0,q,wq)
        call chebyshev_u(this%para%nz,z,wz)
        do a=1,4; do b=1,4
            T(a,b) = merge(One,Zero,a==b)-K(a)*K(b)/(-m**2)
        end do; end do
        val = Zero
        do i=1,this%para%nq
            do j=1,this%para%nz
                qv4 = [Zero,Zero,dcmplx(q(i)*sqrt(1-z(j)**2)),dcmplx(q(i)*z(j))]
                w = wq(i)*q(i)**3*wz(j)*4*pi/(2*pi)**4
                F2 = Famp(ch,dot4(qv4,qv4))**2
                Sp = propagator(qv4+Qv/2)
                SmT = transpose(propagator(-qv4+Qv/2))
                if(ch=='s')then
                    val = val+w*F2*tr2(bar_s,matmul(Sp,matmul(gam_s,SmT)))
                else
                    do a=1,4; do b=1,4
                        val = val+w*F2*T(a,b)*tr2(bar_a(:,:,a),matmul(Sp,matmul(gam_a(:,:,b),SmT)))
                    end do; end do
                end if
            end do
        end do
    end function

    subroutine diquark_equation_solve(this)
        class(diquark_equation):: this
        complex(8):: d
        real(8):: h,cf
        h = this%para%h
        cf = 2d0                                       !Tr_F[t^T t] Tr_C[H^T H] = 1 x 2
        d = (loop(this,'s',1+h)-loop(this,'s',1-h))/(2*h)*cf
        this%g2s = 2*(-this%para%m0**2)/d
        d = (loop(this,'a',1+h)-loop(this,'a',1-h))/(2*h)*cf
        this%g2a = 2*(-this%para%m1**2)*3/d
        this%gs = sqrt(dcmplx(real(this%g2s),0d0))
        this%ga = sqrt(dcmplx(real(this%g2a),0d0))
        if(jrank==0)then
            print '(a,2f12.6,a,2f10.6)','   0+: g^2 =',this%g2s,'   g =',this%gs
            print '(a,2f12.6,a,2f10.6)','   1+: g^2 =',this%g2a,'   g =',this%ga
        end if
    end subroutine

    subroutine diquark_equation_output(this)
        class(diquark_equation):: this
        complex(8):: gs,ga
        namelist /diquark_coupling_list/ gs,ga
        if(jrank/=0) return
        gs = this%gs; ga = this%ga
        call x_mkdir('outdata')
        open(unit=101,file='outdata/'//trim(this%para%root)//'.txt',status='replace')
        write(101,nml=diquark_coupling_list)
        close(101)
    end subroutine

end module

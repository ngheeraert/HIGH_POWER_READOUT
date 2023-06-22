MODULE fortran_module

	implicit none

	real(8),   parameter :: pi=(4._8)*atan(1._8)
	real(8),   parameter :: twopi=2.0_8*(4.0_8)*atan(1.0_8)
	real(8),   parameter :: one=1.0_8
	real(8),   parameter :: zero=0.0_8
	complex(8),parameter :: ic=(0.0_8,1.0_8)

CONTAINS

	SUBROUTINE time_evolve( err_lim, w_qb, wd, A_d, t_in, tref, t_f, dt_in,  wwk, gk, ovm_in, bigW_in, bigL_in, &
		bigU_in, Omat, p_in, f_in, adaptive, lim_slowfactor, p, f, t, max_slow_factor, info )
		real(8), intent(in)       :: 	w_qb(:)
		real(8), intent(in)       :: 	wwk(:)
		real(8), intent(in)       :: 	gk(:,:,:)
		complex(8), intent(in)    :: 	ovm_in(:,:,:,:)
		complex(8), intent(in)    :: 	bigL_in(:,:,:,:)
		complex(8), intent(in)    :: 	bigW_in(:,:,:), bigU_in(:,:,:)
		complex(8), intent(in)    :: 	f_in(:,:,:)
		complex(8), intent(in)    :: 	p_in(:,:)
		real(8), intent(in)       :: 	Omat(:,:)
		real(8), intent(in)       :: 	wd, A_d, t_in, tref, dt_in, t_f, err_lim
		logical, intent(in)       :: 	adaptive
		integer, intent(in)		  ::    lim_slowfactor
		complex(8), dimension(size(p_in,1),size(p_in,2)), intent(out)   :: 	p
		complex(8), dimension(size(p_in,1),size(p_in,2))   :: 	kp1, kp2, kp3, kp4, kp3_RK3
		complex(8), dimension(size(f_in,1), size(f_in,2), size(f_in,3)), intent(out)   ::  f
		complex(8), dimension(size(f_in,1), size(f_in,2), size(f_in,3))::  kf1, kf2, kf3, kf4, kf3_RK3
		real(8), intent(out)      :: 	t
		integer, intent(out)  	  :: 	max_slow_factor
		integer, intent(out)      :: 	info
		real(8)      			  :: 	ot, mid_t, dt, error, t_corrected
		complex(8)			      :: 	ovm(size(p_in,1),size(p_in,2),size(p_in,1),size(p_in,2))
		complex(8)			      :: 	bigL(size(p_in,1),size(p_in,2),size(p_in,1),size(p_in,2))
		complex(8)			      :: 	bigW(size(p_in,1),size(p_in,2),size(p_in,2))
		complex(8)			      :: 	bigU(size(p_in,1),size(p_in,2),size(p_in,2))
		complex(8), dimension(size(p_in,1),size(p_in,2))             		::  mid_p, pdot, op, p_RK3, p_RK4
		complex(8), dimension(size(f_in,1), size(f_in,2), size(f_in,3))     ::  mid_f, fdot, of, f_RK3, f_RK4
		integer                   :: 	nmodes, nq, ncs, slow_factor

		nq = size( f_in, 1 )
		ncs = size( f_in, 2 )
		nmodes= size( f_in, 3 )

		pdot = 0._8
		fdot = 0._8
		mid_p = 0._8
		mid_f = 0._8
		ovm = ovm_in
		bigL = bigL_in
		bigW = bigW_in
		bigU = bigU_in
		slow_factor = 1
		max_slow_factor = 1
		t_corrected = -1.0e9_8
		info=0

		t = t_in
		p = p_in
		f = f_in

		WHILE_DO: DO

			IF ( t > t_f ) EXIT WHILE_DO

			dt = dt_in/slow_factor

			if ( abs(t-tref)<dt_in/200 ) then
				dt = dt/10000
			elseif ( abs(t-tref)<dt_in/20 ) then
				dt = dt/1000
			elseif ( abs(t-tref)<dt_in/2 ) then
				dt = dt/100
			elseif ( abs(t-tref)<5*dt_in ) then
				dt = dt/10
			end if

			ot = t
			of = f
			op = p

			CALL calc_derivatives( w_qb, wd, A_d, t, wwk, gk, ovm, bigW, bigL, bigU, Omat, p, f, pdot, fdot )

			kp1 = pdot
			kf1 = fdot

			!== SECOND STEP OF RK4 ==========
			mid_f = of + 0.5_8*dt*kf1
			mid_p = op + 0.5_8*dt*kp1
			mid_t = ot + 0.5_8*dt
			CALL update_sums( wwk, gk, Omat, mid_f, ovm, bigL, bigW, bigU )

			CALL calc_derivatives( w_qb, wd, A_d, mid_t, wwk, gk, ovm, bigW, bigL, bigU, Omat, mid_p, mid_f, pdot, fdot )

			kp2 = pdot
			kf2 = fdot

			!== THIRD STEP OF RK4 ==========
			mid_f = of + 0.5_8*dt*kf2
			mid_p = op + 0.5_8*dt*kp2
			CALL update_sums( wwk, gk, Omat, mid_f, ovm, bigL, bigW, bigU )

			CALL calc_derivatives( w_qb, wd, A_d, mid_t, wwk, gk, ovm, bigW, bigL, bigU, Omat, mid_p, mid_f, pdot, fdot)

			kp3 = pdot
			kf3 = fdot

			!== FOURTH STEP OF RK4 ==========
			mid_f = of + dt*kf3
			mid_p = op + dt*kp3
			mid_t = ot + dt
			CALL update_sums( wwk, gk, Omat, mid_f, ovm, bigL, bigW, bigU )

			CALL calc_derivatives( w_qb, wd, A_d, mid_t, wwk, gk, ovm, bigW, bigL, bigU, Omat, mid_p, mid_f, pdot, fdot)

			kp4 = pdot
			kf4 = fdot

			!== THIRD STEP OF RK3 ==========
			mid_f = of - dt*kf1 + 2.0_8*dt*kf2
			mid_p = op - dt*kp1 + 2.0_8*dt*kp2
			CALL update_sums( wwk, gk, Omat, mid_f, ovm, bigL, bigW, bigU )

			CALL calc_derivatives( w_qb, wd, A_d, mid_t, wwk, gk, ovm, bigW, bigL, bigU, Omat, mid_p, mid_f, pdot, fdot)

			kp3_RK3 = pdot
			kf3_RK3 = fdot

			p_RK3 = p + dt*( kp1 + 4.0_8*kp2 + kp3_RK3 )/6.0_8
			f_RK3 = f + dt*( kf1 + 4.0_8*kf2 + kf3_RK3 )/6.0_8

			p_RK4 = p + dt*( kp1 + 2.0_8*kp2 + 2.0_8*kp3 + kp4 )/6.0_8
			f_RK4 = f + dt*( kf1 + 2.0_8*kf2 + 2.0_8*kf3 + kf4 )/6.0_8

			error = sum( abs(f_RK4-f_RK3) ) + sum( abs(p_RK4-p_RK3) )

			if ( ( error < err_lim ) .or. (adaptive .eqv. .false.) ) then

				p = p_RK4
				f = f_RK4
				t = t + dt

				if ( (slow_factor > 2) .and. (t-t_corrected > 50*dt) ) then
					if (slow_factor > 8) then
						slow_factor = slow_factor/9
					else
						slow_factor = slow_factor/3
					endif
				end if

			else 

				t_corrected = t
				slow_factor = slow_factor*3
				if ( slow_factor > lim_slowfactor ) then
					info=-1
					print*, 'slow_factor > lim_slowfactor: ABORT'
					stop
				end if
				if (slow_factor > max_slow_factor) then
					max_slow_factor = slow_factor
				end if

			end if

			CALL update_sums( wwk, gk, Omat, f, ovm, bigL, bigW, bigU )

		END DO WHILE_DO
		info = 1

		RETURN
	END SUBROUTINE time_evolve

	SUBROUTINE calc_derivatives( w_qb, wd, A_d, t, wwk, gk, ovm, bigW, bigL, bigU, Omat, p_in, f_in, pdot, fdot )
		real(8), intent(in)       :: w_qb(:)
		real(8), intent(in)       :: wwk(:)
		real(8), intent(in)       :: gk(:,:,:)
		complex(8), intent(in)    :: ovm(:,:,:,:)
		complex(8), intent(in)    :: bigL(:,:,:,:)
		complex(8), intent(in)    :: bigW(:,:,:), bigU(:,:,:)
		complex(8), intent(in)    :: p_in(:,:)
		real(8), intent(in)       :: Omat(:,:)
		complex(8), intent(in)    :: f_in(:,:,:)
		real(8), intent(in)       :: wd, A_d, t
		complex(8)                :: inv_ovm( size(p_in,2), size(p_in,2) )
		complex(8), intent(out)   :: pdot(size(p_in,1),size(p_in,2))
		complex(8), intent(out)   :: fdot(size(f_in,1),size(f_in,2),size(f_in,3))
		integer                   :: nmodes, nq, ncs
		complex(8), dimension(size(f_in,2))             ::  p
		complex(8), dimension(size(f_in,2))             :: bigP
		complex(8), dimension(size(f_in,2), size(f_in,3))     ::  f
		complex(8), dimension(size(f_in,2), size(f_in,3))     :: bigF
		complex(8), dimension(size(f_in,2))             :: dE_dpc_sj_ST
		complex(8), dimension(size(f_in,2), size(f_in,3))     :: dE_dyc_sj_ST
		complex(8), dimension(size(f_in,2),size(f_in,2))         :: inv_ov_ff, ov_ff
		complex(8), dimension(size(f_in,2),size(f_in,2))         ::  b, Amat
		complex(8), dimension(size(f_in,2),size(f_in,2))    :: RHS
		complex(8), allocatable          :: packed_RHS(:), d_packed(:)
		complex(8), dimension(size(f_in,2),size(f_in,2))        :: d
		complex(8), dimension(size(f_in,2), size(f_in,3))     :: inv_ov_ff_mul_F_m_fnP
		complex(8), dimension(size(f_in,2),size(f_in,2),size(f_in,2))  :: alphaT
		complex(8), allocatable  :: mat2D(:,:)
		integer                                :: info,i,j,k,n,m,s

		allocate( mat2D( size(f_in,2)**2,size(f_in,2)**2 ) )
		allocate( packed_RHS( size(f_in,2)**2 ) )
		allocate( d_packed( size(f_in,2)**2 ) )

		nq = size( f_in, 1 )
		ncs = size( f_in, 2 )
		nmodes= size( f_in, 3 )

		do s=1,nq

			mat2D = 0._8
			packed_RHS = 0._8
			d_packed = 0._8

			p = p_in(s,:)
			f = f_in(s,:,:)

			ov_ff=ovm(s,:,s,:)
			inv_ov_ff=ov_ff
			CALL invertH(ncs,inv_ov_ff,info)
			inv_ovm = inv_ov_ff

			do n=1,ncs
				call dE_dpc_sj( w_qb, wd, A_d, t, p_in, ovm, bigL, bigW, bigU, s, &
					n, dE_dpc_sj_ST(n))
				call dE_dyc_sjk( wwk, wd, A_d, t, gk, w_qb, p_in, f_in, ovm, bigL, bigW, &
					bigU, Omat, s, n, dE_dyc_sj_ST(n,:) )
			end do

			bigP = -Ic*dE_dpc_sj_ST
			do k=1, nmodes
				bigF(:,k) = -Ic*( dE_dyc_sj_ST(:,k)/conjg(p)&
					+0.5_8*( dE_dpc_sj_ST + conjg(dE_dpc_sj_ST)*p/conjg(p) )*f(:,k) )
			end do
			inv_ov_ff_mul_F_m_fnP = 0._8
			do n=1, ncs
				do k=1, nmodes
					inv_ov_ff_mul_F_m_fnP(n,k) = sum( inv_ov_ff(n,:)*(bigF(:,k)-f(n,k)*bigP(:)) )
					!inv_ov_ff_mul_F_m_fnP(n,k) = sum( inv_ov_ff(n,:)*( bigF(:,k) )
					!inv_ov_ff_mul_F_m_fnP(n,k) = inv_ov_ff_mul_F_m_fnP(n,k) &
					!    - sum( inv_ov_ff(n,:)*f(n,k)*bigP(:) )
				end do
			end do
			Amat = matmul( conjg(f(:,:)), TRANSPOSE(inv_ov_ff_mul_F_m_fnP) )
			b(:,:) = matmul( conjg(f(:,:)),TRANSPOSE(f(:,:)) )	
			!
			!
			!-- build alphaTensor
			do i=1, ncs
				do n=1, ncs
					do m=1, ncs
						alphaT(i,n,m)=sum(inv_ov_ff(i,:)*ov_ff(:,n)*(b(:,m)-b(:,n)) )
					end do
				end do
			end do
			!
			RHS = matmul(inv_ov_ff, ov_ff * Amat)
			do i=1, ncs
				do n=1, ncs
					packed_RHS((n-1)*ncs+i)=RHS(i,n)
				end do
			end do
			!
			do i=1, ncs
				do n=1, ncs
					do j=1, ncs
						do m=1, ncs
							mat2D((n-1)*ncs+i,(m-1)*ncs+j) =  KD(m,n)*KD(i,j) 
							mat2D((n-1)*ncs+i,(m-1)*ncs+j) = &
								mat2D((n-1)*ncs+i,(m-1)*ncs+j) + alphaT(i,n,m)*KD(j,n)
						end do
					end do
				end do
			end do
			!
			CALL solveEq_c( mat2D , packed_RHS, d_packed )
			!
			do i=1, ncs
				do n=1, ncs
					d(i,n) = d_packed((n-1)*ncs+i)  !! why /2 ????
				end do
			end do
			!
			do n=1, ncs
				do k=1, nmodes
					fdot(s,n,k) = ( inv_ov_ff_mul_F_m_fnP(n,k) - sum( d(n,:)*(f(:,k)-f(n,k)) ) )/p(n)
				end do
				!
				pdot(s,n) = sum( inv_ov_ff(n,:)* bigP(:) ) - sum( d(n,:) ) &
					+ 0.5_8*p(n)*( sum( fdot(s,n,:)*conjg(f(n,:)) &
					+ conjg(fdot(s,n,:))*f(n,:)) )
			end do

		end do

		return
	END SUBROUTINE

	SUBROUTINE get_error( w_qb, wd, ad, t, new_t, wwk, gk, ovm, bigW, bigL, &
			bigU, Omat, g2mat, p, y, pdot, ydot, new_pdot, new_ydot, energy_t, error )

		real(8), intent(in)       ::	 w_qb(:)
		real(8), intent(in)       ::	 wwk(:)
		real(8), intent(in)       ::	 gk(:,:,:)
		real(8), intent(in)       ::	 Omat(:,:)
		complex(8), intent(in)    ::	 bigL(:,:,:,:)
		complex(8), intent(in)    ::	 bigW(:,:,:), bigU(:,:,:)
		complex(8), intent(in)    ::	 g2mat(:,:,:,:), ovm(:,:,:,:)
		complex(8), intent(in)    ::	 p(:,:), pdot(:,:), new_pdot(:,:)
		complex(8), intent(in)    ::	 y(:,:,:), ydot(:,:,:), new_ydot(:,:,:)
		real(8), intent(in)       ::	 wd, ad, t, new_t, energy_t
		real(8), intent(out)   	  ::	 error
		complex(8)				  ::	 tmp1, tmp2, tmp3, tmp4, tmp_sum, sum_diag_g2mat
		real(8)				  	  ::	 at
		integer                   ::	 nmodes, nl, ncs, s,m,l,n,i,j,k,pp
		complex(8), dimension(size(p,1),size(p,2))         				::	 pc, pdotc
		complex(8), dimension(size(y,1),size(y,2),size(y,3))     		::	 yc, ydotc
		complex(8), dimension(size(p,2))         						::	 opdd
		complex(8), dimension(size(y,2),size(y,3))     					::	 oydd
		complex(8), dimension(size(p,2),size(p,2))         			   	::	 ovmr
		complex(8), dimension(size(p,1),size(p,2),size(p,1),size(p,2)) 	::	 kap

		tmp1 = 0._8
		tmp2 = 0._8
		tmp3 = 0._8
		tmp4 = 0._8

		pc = conjg( p )
		yc = conjg( y )
		pdotc = conjg( pdot )
		ydotc = conjg( ydot )

		at = ad*cos( wd*t )

		nl = size( p,1 )
		ncs = size( p,2 )
		nmodes = size( y,3 )

		do s=1, nl
			do m=1, ncs
				do l=1, nl
					do n=1, ncs
						kap( s,m,l,n ) = sum( ydot(s,m,:)*yc(s,m,:) &
						+ ydotc(s,m,:)*y(s,m,:) - 2._8*yc(l,n,:)*ydot(s,m,:) ) 
					end do
				end do
			end do
		end do

		LOOPi: do i=1, nl

			ovmr = ovm(i,:,i,:)

			opdd = ( new_pdot(i,:) - pdot(i,:) )/(new_t-t)
			oydd = ( new_ydot(i,:,:) - ydot(i,:,:) )/(new_t-t)

			do m=1, ncs
				do n=1, ncs
					!==== tmp1 cajcujation
					tmp1 = tmp1 + ovmr(m,n)*( &
							+ pdotc(i,m)*pdot(i,n) &
							- 0.5_8 * pdotc(i,m)*p(i,n)*kap(i,n,i,m) &
							- 0.5_8 * pc(i,m)*pdot(i,n)*conjg(kap(i,m,i,n)) &
							+ pc(i,m)*p(i,n)*( sum( ydotc(i,m,:)*ydot(i,n,:) )&
							+ 0.25_8*conjg(kap(i,m,i,n))*kap(i,n,i,m)&
							)&
							)

					!==== tmp4 cajcujation
					tmp4 = tmp4 + pc(i,m)*ovmr(m,n)*( &
							+ opdd(n) &
							- pdot(i,n)*kap(i,n,i,m) &
							+ p(i,n)*( sum( yc(i,m,:)*oydd(n,:)&
							- 0.5_8*( y(i,n,:)*conjg(oydd(n,:))&
							+ yc(i,n,:)*oydd(n,:)&
							+ 2._8*ydotc(i,n,:)*ydot(i,n,:) ) )&
							+ 0.25_8*kap(i,n,i,m)**2 )&
							)

					!==== tmp2 cajcujation
					tmp2 = tmp2 + pdotc(i,m)*p(i,n)*ovm(i,m,i,n)*( &
							+ w_qb(i) &
							+ bigW(i,m,n)  &
							+ at*bigU(i,m,n) ) &
							+ pc(i,m)*p(i,n)*ovmr(m,n)*( &
							- 0.5_8*conjg(kap(i,m,i,n)) &
							*( w_qb(i)+bigW(i,m,n)+at*bigU(i,m,n) ) &
							+ sum( wwk(:)*ydotc(i,m,:)*y(i,n,:) &
							+ at*Omat(1,:)*ydotc(i,m,:) ) ) 

					do j=1, nl
						tmp2 = tmp2 + pdotc(i,m)*p(j,n)*ovm(i,m,j,n)*bigL(i,m,j,n) &
								+ pc(i,m)*p(j,n)*ovm(i,m,j,n)*( &
								- 0.5_8*conjg(kap(i,m,j,n))*bigL(i,m,j,n)  &
								+ sum( gk(i,j,:)*ydotc(i,m,:) ) )
					end do

					!==== tmp3 cajcujation
					tmp3 = tmp3 + pc(i,m)*p(i,n)*ovm(i,m,i,n)*( (w_qb(i) &
							+ bigW(i,m,n) &
							+ at*bigU(i,m,n) )**2 &
							+ sum( wwk(:)**2*yc(i,m,:)*y(i,n,:)  &
							+ wwk(:)*at*Omat(1,:)*(yc(i,m,:)+y(i,n,:)) &
							+ at**2*(Omat(1,:))**2 ) )

					do j=1, nl
						tmp_sum = 0._8
						do pp=1, nmodes
							tmp_sum = tmp_sum + ( yc(i,m,pp) + y(j,n,pp) ) &
									* sum( ( g2mat(j,i,pp,:) &
									+ 2._8*at*Omat(1,pp)*gk(i,j,:) ) &
									* ( yc(i,m,:) + y(j,n,:) ) )
						end do

						sum_diag_g2mat = 0._8
						do k=1, nmodes
							sum_diag_g2mat = sum_diag_g2mat + g2mat(i,j,k,k)
						end do
							
						tmp3 = tmp3 + pc(i,m)*p(j,n)*ovm(i,m,j,n)*( tmp_sum &
								+ sum_diag_g2mat &
								+ 2._8*at*sum( Omat(1,:)*gk(i,j,:) ) &
								+ ( w_qb(i)+w_qb(j) )*bigL(i,m,j,n) &
								+ 2._8*sum( wwk*yc(i,m,:)*y(j,n,:) )*bigL(i,m,j,n) &
								+ sum( gk(i,j,:)&
								*wwk(:)*( y(j,n,:)+yc(i,m,:) ) ) )
					end do
				end do
			end do

		end do LOOPi

		error = ( -0.5*real(tmp4) + 0.5*tmp1 - 2*aimag(tmp2) + tmp3 ) / ( (energy_t)**2 )

		return
	END SUBROUTINE

	SUBROUTINE update_sums( wwk, gk, Omat, f_in, ovm, bigL, bigW, bigU )
		real(8), intent(in)                :: wwk(:)
		real(8), intent(in)                :: gk(:,:,:)
		real(8), intent(in)                :: Omat(:,:)
		complex(8), intent(in)             :: f_in(:,:,:)
		complex(8),dimension( size(f_in,1),size(f_in,2),size(f_in,1),size(f_in,2) ),&
			intent(out)    ::  bigL, ovm
		complex(8),dimension( size(f_in,1),size(f_in,2),size(f_in,2) ),&
			intent(out)    ::  bigW, bigU
		integer ::  m,n,i,j


		bigW = 0._8
		do m=1, size(f_in,2)
			do n=1, size(f_in,2)
				do i=1, size(f_in,1)

					bigW(i,m,n) = dot_product( wwk(:)*f_in(i,m,:), f_in(i,n,:) )
					bigU(i,m,n) = sum( Omat(1,:) * ( conjg(f_in(i,m,:)) + f_in(i,n,:) ) )

					do j=1, size(f_in,1)
						CALL calc_overlap( f_in(i,m,:), f_in(j,n,:), ovm(i,m,j,n) )
						bigL(i,m,j,n) = &
							sum( gk(i,j,:) * (conjg(f_in(i,m,:)) + f_in(j,n,:)) )
					end do

				end do
			end do
		end do

		return
	END SUBROUTINE

	SUBROUTINE calc_overlap( f1, f2, overlap )
		complex(8), intent(in)   ::  f1( : ), f2( : )
		complex(8), intent(out)  ::   overlap
		!= internal variables
		complex(8)    :: tmp1, tmp2, tmp3

		if (size(f1,1) .ne. 1) then
			tmp1 = dot_product(f1, f1)
			tmp2 = dot_product(f2, f2)
			tmp3 = dot_product(f1, f2)
		else
			tmp1 = conjg(f1(1))*f1(1)
			tmp2 = conjg(f2(1))*f2(1)
			tmp3 = conjg(f1(1))*f2(1)
		end if

		overlap = exp( -0.5_8*tmp1 - 0.5_8*tmp2 + tmp3 ) 

		return
	END SUBROUTINE

	SUBROUTINE dE_dpc_sj(w_qb, wd, A_d, t, p_in, ov_ff, bigL, bigW, bigU, s, j, outv )
		complex(8), intent(in)                                  ::  p_in(:,:)
		complex(8), intent(in)                                  ::  ov_ff(:,:,:,:)
		complex(8), intent(in)                                  ::  bigL(:,:,:,:)
		complex(8), intent(in)                                  ::  bigW(:,:,:)
		complex(8), intent(in)                                  ::  bigU(:,:,:)
		real(8), intent(in)         				            ::  w_qb(:)   
		real(8), intent(in)         				            ::  t, wd, A_d
		integer,intent(in)              			 			::  s,j     
		complex(8), intent(out) 	       		 				::  outv
		integer													::  i


		outv = 0._8

		outv = sum( p_in(s,:)*ov_ff(s,j,s,:)*( w_qb(s) + bigW(s,j,:) &
			+ A_d*cos( wd * t ) * bigU(s,j,:)  )) 
		do i=1, size(p_in,1)
			outv = outv + sum( p_in(i,:)*ov_ff(s,j,i,:)*bigL(s,j,i,:) )
		end do

		return
	END SUBROUTINE

	SUBROUTINE dE_dpc_sj_qtr(w_qb, g,  wd, A_d, t, p_in, y0, ov_ff, bigL, bigW, s, j, outv )
		complex(8), intent(in)                                  ::  p_in(:,:)
		real(8), intent(in)                                  ::  g(:,:)
		complex(8), intent(in)                                  ::  y0(:,:)
		complex(8), intent(in)                                  ::  ov_ff(:,:,:,:)
		complex(8), intent(in)                                  ::  bigL(:,:,:)
		complex(8), intent(in)                                  ::  bigW(:,:,:)
		real(8), intent(in)         				            ::  w_qb(:)   
		real(8), intent(in)         				            ::  t, wd, A_d
		integer,intent(in)              			 			::  s,j     
		complex(8), intent(out) 	       		 				::  outv
		integer													::  i
		complex(8), dimension( size(y0,1),size(y0,2) )  ::  y0c


		y0c = conjg(y0)

		outv = 0._8

		outv = sum( p_in(s,:)*ov_ff(s,j,s,:)*( w_qb(s) + bigW(s,j,:) &
			+ ( y0c(s,j) + y0(s,:) ) * bigL( s, j, : ) &
			+ A_d*cos( wd * t ) * ( y0c(s,j) + y0(s,:) )  ) )
		do i=1, size(p_in,1)
			outv = outv + sum( g(s,i)*p_in(i,:)*ov_ff(s,j,i,:)*( &
				y0c(s,j)**2 + y0(i,:)**2 + 2._8*y0c(s,j)*y0(i,:) + 1._8 ) )
		end do

		return
	END SUBROUTINE

	SUBROUTINE dE_dyc_sj0_qtr(w_cav, wd, A_d, t, g, w_qb, p, y0, ov_ff, bigL, bigW, s, j,  outv )
		real(8), intent(in)                                  ::  w_cav
		real(8), intent(in)                                  ::  g(:,:)
		complex(8), intent(in)                                  ::  p(:,:)
		complex(8), intent(in)                                  ::  y0(:,:)
		complex(8), intent(in)                                  ::  ov_ff(:,:,:,:)
		complex(8), intent(in)                                  ::  bigL(:,:,:)
		complex(8), intent(in)                                  ::  bigW(:,:,:)
		real(8), intent(in)         				            ::  w_qb(:)   
		real(8), intent(in)         				            ::  t, wd, A_d
		integer,intent(in)              			 			::  s,j
		complex(8),intent(out) 	         ::  outv
		complex(8), dimension( size(p,1),size(p,2) )            ::  pc
		complex(8), dimension( size(y0,1),size(y0,2) )  ::  y0c
		integer ::  n,i

		pc = conjg( p )
		y0c = conjg( y0 )

		outv=0._8

		do n=1, size( p,2 )
			outv = outv + pc(s,j)*p(s,n)*ov_ff(s,j,s,n)*( w_cav*y0(s,n) &
				+ bigL(s,j,n) + A_d*cos( wd*t ) &
				+ (y0(s,n)-0.5_8*y0(s,j))*( w_qb(s) + bigW(s,j,n) &
				+ (y0c(s,j)+y0(s,n))*bigL(s,j,n) + A_d*cos( wd*t )*( y0c(s,j) + y0(s,n) )  ) &
				) &
				- 0.5_8*pc(s,n)*p(s,j)*ov_ff(s,n,s,j)*y0(s,j)*( w_qb(s) &
				+ bigW(s,n,j) + (y0c(s,n)+y0(s,j))*bigL(s,n,j) + A_d*cos( wd*t )*( y0c(s,n) + y0(s,j) ) )

			do i=1, size( p,1 )
				outv = outv + g(s,i)*pc(s,j)*p(i,n)*ov_ff(s,j,i,n)*( &
					- 0.5_8*(y0(s,j)-2._8*y0(i,n))&
					*( y0c(s,j)**2 + y0(i,n)**2 + 2._8*y0c(s,j)*y0(i,n) + 1._8 ) &
					+ 2._8*( y0c(s,j) + y0(i,n) ) )  &
					- 0.5_8 * g(i,s)*pc(i,n)*p(s,j)*ov_ff(i,n,s,j)*y0(s,j)&
					* ( y0c(i,n)**2 + y0(s,j)**2 + 2._8*y0c(i,n)*y0(s,j) + 1._8 )
			end do

		end do

		return
	END SUBROUTINE

	SUBROUTINE dE_dyc_sjk_qtr(wwk, wd, A_d, t, g, gk, w_qb, p_in, y, ov_ff, bigL, bigW, s, j,  outv )
		real(8), intent(in)                                  ::  wwk(:)
		real(8), intent(in)                                  ::  g(:,:)
		real(8), intent(in)                                  ::  gk(:)
		complex(8), intent(in)                                  ::  p_in(:,:)
		complex(8), intent(in)                                  ::  y(:,:,:)
		complex(8), intent(in)                                  ::  ov_ff(:,:,:,:)
		complex(8), intent(in)                                  ::  bigL(:,:,:)
		complex(8), intent(in)                                  ::  bigW(:,:,:)
		real(8), intent(in)         				            ::  w_qb(:)   
		real(8), intent(in)         				            ::  t, wd, A_d
		integer,intent(in)              			 			::  s,j
		complex(8), dimension( size(gk,1) ), intent(out) 	        ::  outv(:)
		complex(8), dimension( size(p_in,1),size(p_in,2) )            ::  p, pc
		complex(8), dimension( size(y,1),size(y,2),size(y,3) )  ::  yc
		integer ::  n,i

		p = p_in
		pc = conjg( p )
		yc = conjg( y )

		outv=0._8

		do n=1, size( p,2 )
			outv = outv + pc(s,j)*p(s,n)*ov_ff(s,j,s,n)*( wwk(2:)*y(s,n,2:) &
				+ gk(:)*( yc(s,j,1)+y(s,n,1) ) &
				+ (y(s,n,2:)-0.5_8*y(s,j,2:))*( w_qb(s) + bigW(s,j,n) &
				+ (yc(s,j,1)+y(s,n,1))*bigL(s,j,n) + A_d*cos( wd*t )*( yc(s,j,1) + y(s,n,1) ) ) &
				) &
				- 0.5_8*pc(s,n)*p(s,j)*ov_ff(s,n,s,j)*y(s,j,2:)*( w_qb(s) &
				+ bigW(s,n,j) + (yc(s,n,1)+y(s,j,1))*bigL(s,n,j) + A_d*cos( wd*t )*( yc(s,n,1) + y(s,j,1) ) )

			do i=1, size( p,1 )
				outv = outv + g(s,i)*pc(s,j)*p(i,n)*ov_ff(s,j,i,n)*( &
					- 0.5_8*(y(s,j,2:)-2._8*y(i,n,2:))&
					*( yc(s,j,1)**2 + y(i,n,1)**2 + 2._8*yc(s,j,1)*y(i,n,1) + 1._8 ) ) &
					- 0.5_8 * g(i,s)*pc(i,n)*p(s,j)*ov_ff(i,n,s,j)*y(s,j,2:)&
					* ( yc(i,n,1)**2 + y(s,j,1)**2 + 2._8*yc(i,n,1)*y(s,j,1) + 1._8 )
			end do

		end do

		return
	END SUBROUTINE

	SUBROUTINE dE_dyc_sjk(wwk, wd, A_d, t, gk, w_qb, p_in, f_in, ov_ff, bigL, bigW, bigU,&
			Omat, s, j,  outv )

		real(8), intent(in)                                  ::  wwk(:)
		real(8), intent(in)                                  ::  gk(:,:,:)
		complex(8), intent(in)                                  ::  p_in(:,:)
		complex(8), intent(in)                                  ::  f_in(:,:,:)
		complex(8), intent(in)                                  ::  ov_ff(:,:,:,:)
		complex(8), intent(in)                                  ::  bigL(:,:,:,:)
		complex(8), intent(in)                                  ::  bigW(:,:,:)
		complex(8), intent(in)                                  ::  bigU(:,:,:)
		real(8), intent(in)                                     ::  Omat(:,:)
		real(8), intent(in)         				            ::  w_qb(:)   
		real(8), intent(in)         				            ::  t, wd, A_d
		integer,intent(in)              			 			::  s,j
		complex(8),dimension(size(wwk,1)), intent(out) 	         ::  outv(:)
		complex(8), dimension( size(p_in,1),size(p_in,2) )            ::  p, pc
		complex(8), dimension( size(f_in,1),size(f_in,2),size(f_in,3) )  ::  y
		integer ::  n,i

		p = p_in
		pc = conjg(p)
		y = f_in

		outv=0._8

		do n=1, size( p,2 )
			outv = outv + pc(s,j)*p(s,n)*ov_ff(s,j,s,n)*( wwk(:)*y(s,n,:) &
				+ A_d*cos( wd*t )*Omat(1,:)  &
				+(y(s,n,:)-0.5_8*y(s,j,:))*( w_qb(s)+bigW(s,j,n) &
				+ A_d*cos(wd*t)*bigU(s,j,n) ) ) &
				- 0.5_8*pc(s,n)*p(s,j)*ov_ff(s,n,s,j)*y(s,j,:)*( w_qb(s) &
				+ bigW(s,n,j) &
				+ A_d*cos(wd*t)*bigU(s,n,j) )

			do i=1, size( p,1 )
				outv = outv + pc(s,j)*p(i,n)*ov_ff(s,j,i,n)*( &
					+ gk(s,i,:) &
					+ ( y(i,n,:)-0.5_8*y(s,j,:) )*bigL(s,j,i,n) ) &
					- 0.5_8*pc(i,n)*p(s,j)*ov_ff(i,n,s,j) &
					*y(s,j,:)*bigL(i,n,s,j)
			end do

		end do

		return
	END SUBROUTINE

	SUBROUTINE calc_wigner( p_in, f_in, ovm, xmin, xmax,  wigner, xnum )
		complex(8), intent(in)                          ::  p_in(:,:)
		complex(8), intent(in)                          ::  f_in(:,:)
		integer, intent(in)							    ::  xnum
		real(8), intent(in)							    ::  xmin, xmax
		complex(8), intent(in)                          ::  ovm(:,:,:,:)
		real(8)							                ::  dx, x, p
		complex(8)					  					::  tmp, zz
		integer							  				::  i,n,m,xi,xj,ncs,nl
		real(8), intent(out)				            ::  wigner(xnum,xnum)

		nl = size( p_in, 1 )
		ncs = size( p_in, 2 )
		dx =   (xmax - xmin) / dble(xnum-1) 

		do xi=1, xnum
			do xj=1, xnum

				tmp = 0._8

				x = xmin + dx*(xi-1)
				p = xmin + dx*(xj-1)

				do i=1, nl
					do n=1, ncs
						do m=1, ncs

							zz =  x + Ic*p 

							tmp = tmp + conjg(p_in(i,n))*p_in(i,m) &
								* exp( -2._8 * ( real(zz)+Ic*aimag(zz) - f_in(i,m) ) &
								* ( real(zz)-Ic*aimag(zz) - conjg( f_in(i,n) ) ) ) &
								* ovm(i,n,i,m)

						end do
					end do
				end do

				wigner( xi, xj ) = (2._8/pi)*tmp

			end do
		end do

		return
	END SUBROUTINE

	SUBROUTINE SolveEq_c(A,B,res)
		COMPLEX(8), intent(in)                   ::  A(:,:)
		COMPLEX(8), intent(in)					 ::  B(:)
		COMPLEX(8), intent(out)					 ::  res(size(B,1))
		INTEGER                	   		    	 ::  INFO,LDA,LDB,N,NRHS
		INTEGER, dimension(size(A,1))		  	 ::  IPIV   !-- pivot indices

		NRHS = 1  						!-- number of right hand sides
		N = size(A,1)					!-- the number of linear equations
		LDA = size(A,1)				!-- the leading dimension of A multiply_c
		LDB = size(B,1)				!-- the leading dimension of B
		info=0							!-- 0 is successful

		res = B

		CALL ZGESV(N,NRHS,A,LDA,IPIV,res,LDB,INFO)  !-- Solve by performing the Bunch-Kaufman factorisation

		if (info /= 0) then
			print*, "Failure in DGESV - solving real system of equations"
			print*,"INFO = ",info
		end if
	END SUBROUTINE

	SUBROUTINE InvertH(ncs,A,info)
		integer, intent(in)								 ::  ncs
		COMPLEX(8), intent(in out)                  ::  A(ncs,ncs)
		integer, intent(out)								 ::  info
		INTEGER                	   		          ::  LDA,N,LWORK,i,j
		INTEGER, dimension(size(A,1))		          ::  IPIV
		COMPLEX(8), allocatable 				          ::  WORK(:)

		LDA = size(A,1)
		N = size(A,2)
		info=0
		LWORK = N
		allocate(WORK(LWORK))

		CALL ZHETRF('U',N,A,LDA,IPIV,WORK,LWORK,INFO)  !-- Performs the Bunch-Kaufman factorisation

		if (info==0) then

			CALL ZHETRI('U',N,A,LDA,IPIV,WORK,INFO) !-- CAREFUL: returns only triangular part
			do i=1,N
				do j=1,N
					if (i>j) then
						a(i,j) = conjg(a(j,i))
					end if
				end do
			end do

			if (info /= 0) then
				print*, "Failure in the inversion step, ZHETRI"
				print*,"info=", info
			end if

		else
			print*, "Failure in ZHETRF, Bunch-Kaufman factorisation"
			print*,"info=", info
		end if
	END SUBROUTINE InvertH

	FUNCTION KD(a,b)
		integer,intent(in)   ::  a,b
		integer     ::  KD

		if (a==b) then
			KD = 1
		else
			KD = 0
		end if
	END FUNCTION

	!-- OBSCELETE
	SUBROUTINE time_evolve_qtr( w_qb, wd, A_d, t_in, t_f, dt_in,  wwk, g_qc, gk, ovm_in, bigW_in, bigL_in, p_in, f_in, p, f, t )
		real(8), intent(in)       :: 	w_qb(:)
		real(8), intent(in)       :: 	wwk(:)
		real(8), intent(in)       :: 	gk(:)
		complex(8), intent(in)    :: 	ovm_in(:,:,:,:)
		complex(8), intent(in)    :: 	bigL_in(:,:,:)
		complex(8), intent(in)    :: 	bigW_in(:,:,:)
		complex(8), intent(in)    :: 	f_in(:,:,:)
		complex(8), intent(in)    :: 	p_in(:,:)
		real(8), intent(in)       ::    g_qc(:,:)
		real(8), intent(in)       :: 	wd, A_d, t_in, dt_in, t_f
		complex(8), intent(out)   :: 	p( size(p_in,1),size(p_in,2) )
		complex(8), intent(out)   :: 	f( size(f_in,1),size(f_in,2),size(f_in,3) )
		real(8), intent(out)      :: 	t
		real(8)      			  :: 	ot, mid_t, dt
		complex(8)			      :: 	ovm(size(p_in,1),size(p_in,2),size(p_in,1),size(p_in,2))
		complex(8)			      :: 	bigL(size(p_in,1),size(p_in,2),size(p_in,1))
		complex(8)			      :: 	bigW(size(p_in,1),size(p_in,2),size(p_in,2))
		complex(8), dimension(size(p_in,1),size(p_in,2))             		::  mid_p, pdot, op
		complex(8), dimension(size(f_in,1), size(f_in,2), size(f_in,3))     ::  mid_f, fdot, of
		integer                   :: 	nmodes, nq, ncs

		nq = size( f_in, 1 )
		ncs = size( f_in, 2 )
		nmodes= size( f_in, 3 )

		pdot = 0._8
		fdot = 0._8
		mid_p = 0._8
		mid_f = 0._8
		ovm = ovm_in
		bigL = bigL_in
		bigW = bigW_in

		t = t_in
		p = p_in
		f = f_in

		WHILE_DO: DO

			IF ( t > t_f ) EXIT WHILE_DO

			if ( abs(t)<5*dt_in/1000 ) then
				dt = dt_in/10000
			elseif ( abs(t)<5*dt_in/100 ) then
				dt = dt_in/1000
			elseif ( abs(t)<5*dt_in/10 ) then
				dt = dt_in/100
			elseif ( abs(t)<5*dt_in ) then
				dt = dt_in/10
			elseif ( abs(t)>dt_in ) then
				dt = dt_in
			end if

			ot = t
			of = f
			op = p

			CALL calc_derivatives_qtr( w_qb, wd, A_d, t, wwk, g_qc, gk, ovm, bigW, bigL, p, f, pdot, fdot )

			f = f + fdot*dt/6.0
			p = p + pdot*dt/6.0
			t = t + dt/6.0

			!== SECOND STEP OF RK4 ==========
			mid_f = of + 0.5*dt*fdot
			mid_p = op + 0.5*dt*pdot
			mid_t = ot + 0.5*dt
			CALL update_sums_qtr( wwk, gk, mid_f, ovm, bigL, bigW )

			CALL calc_derivatives_qtr( w_qb, wd, A_d, mid_t, wwk, g_qc, gk, ovm, bigW, bigL, mid_p, mid_f, pdot, fdot )

			f = f + fdot*dt/3.0
			p = p + pdot*dt/3.0
			t = t + dt/3.0

			!== THIRD STEP OF RK4 ==========
			mid_f = of + 0.5*dt*fdot
			mid_p = op + 0.5*dt*pdot
			mid_t = ot + 0.5*dt
			CALL update_sums_qtr( wwk, gk, mid_f, ovm, bigL, bigW )

			CALL calc_derivatives_qtr( w_qb, wd, A_d, mid_t, wwk, g_qc, gk, ovm, bigW, bigL, mid_p, mid_f, pdot, fdot)

			f = f + fdot*dt/3.0
			p = p + pdot*dt/3.0
			t = t + dt/3.0

			!== FOURTH STEP OF RK4 ==========
			mid_f = of + dt*fdot
			mid_p = op + dt*pdot
			mid_t = ot + dt
			CALL update_sums_qtr( wwk, gk, mid_f, ovm, bigL, bigW )

			CALL calc_derivatives_qtr( w_qb, wd, A_d, mid_t, wwk, g_qc, gk, ovm, bigW, bigL, mid_p, mid_f, pdot, fdot)

			f = f + fdot*dt/6.0
			p = p + pdot*dt/6.0
			t = t + dt/6.0
			CALL update_sums_qtr( wwk, gk, f, ovm, bigL, bigW )

		END DO WHILE_DO

		RETURN
	END SUBROUTINE

	SUBROUTINE calc_derivatives_qtr( w_qb, wd, A_d, t, wwk, g, gk, ovm, bigW, bigL, p_in, f_in, pdot, fdot )
		real(8), intent(in)       :: w_qb(:)
		real(8), intent(in)       :: wwk(:)
		real(8), intent(in)       :: g(:,:)
		real(8), intent(in)       :: gk(:)
		complex(8), intent(in)    :: ovm(:,:,:,:)
		complex(8), intent(in)    :: bigL(:,:,:)
		complex(8), intent(in)    :: bigW(:,:,:)
		complex(8), intent(in)    :: p_in(:,:)
		complex(8), intent(in)    :: f_in(:,:,:)
		real(8), intent(in)       :: wd, A_d, t
		complex(8)                :: inv_ovm( size(p_in,2), size(p_in,2) )
		complex(8), intent(out)   :: pdot(size(p_in,1),size(p_in,2))
		complex(8), intent(out)   :: fdot(size(f_in,1),size(f_in,2),size(f_in,3))
		integer                   :: nmodes, nq, ncs
		complex(8), dimension(size(f_in,2))             ::  p
		complex(8), dimension(size(f_in,2))             :: bigP
		complex(8), dimension(size(f_in,2), size(f_in,3))     ::  f
		complex(8), dimension(size(f_in,2), size(f_in,3))     :: bigF
		complex(8), dimension(size(f_in,2))             :: dE_dpc_sj_ST
		complex(8), dimension(size(f_in,2), size(f_in,3))     :: dE_dyc_sj_ST
		complex(8), dimension(size(f_in,2),size(f_in,2))         :: inv_ov_ff, ov_ff
		complex(8), dimension(size(f_in,2),size(f_in,2))         ::  b, Amat
		complex(8), dimension(size(f_in,2),size(f_in,2))    :: RHS
		complex(8), allocatable          :: packed_RHS(:), d_packed(:)
		complex(8), dimension(size(f_in,2),size(f_in,2))        :: d
		complex(8), dimension(size(f_in,2), size(f_in,3))     :: inv_ov_ff_mul_F_m_fnP
		complex(8), dimension(size(f_in,2),size(f_in,2),size(f_in,2))  :: alphaT
		complex(8), allocatable  :: mat2D(:,:)
		integer                                :: info,i,j,k,n,m,s

		allocate( mat2D( size(f_in,2)**2,size(f_in,2)**2 ) )
		allocate( packed_RHS( size(f_in,2)**2 ) )
		allocate( d_packed( size(f_in,2)**2 ) )

		nq = size( f_in, 1 )
		ncs = size( f_in, 2 )
		nmodes= size( f_in, 3 )

		do s=1,nq

			dE_dyc_sj_ST(:,:) = 0._8
			mat2D = 0._8
			packed_RHS = 0._8
			d_packed = 0._8

			p = p_in(s,:)
			f = f_in(s,:,:)

			ov_ff=ovm(s,:,s,:)
			inv_ov_ff=ov_ff
			CALL invertH(ncs,inv_ov_ff,info)
			inv_ovm = inv_ov_ff

			do n=1,ncs
				call dE_dpc_sj_qtr( w_qb, g, wd, A_d, t, p_in, f_in(:,:,1), ovm, bigL, bigW, s, &
					n, dE_dpc_sj_ST(n) )
				call dE_dyc_sj0_qtr( wwk(1), wd, A_d, t, g, w_qb, p_in, f_in(:,:,1), ovm, bigL, bigW, &
					s, n, dE_dyc_sj_ST(n,1) )
				call dE_dyc_sjk_qtr( wwk, wd, A_d, t, g, gk,  w_qb, p_in, f_in, ovm, bigL, bigW, &
					s, n, dE_dyc_sj_ST(n,2:) )
			end do

			bigP = -Ic*dE_dpc_sj_ST
			do k=1, nmodes
				bigF(:,k) = -Ic*( dE_dyc_sj_ST(:,k)/conjg(p)&
					+0.5_8*( dE_dpc_sj_ST + conjg(dE_dpc_sj_ST)*p/conjg(p) )*f(:,k) )
			end do
			inv_ov_ff_mul_F_m_fnP = 0._8
			do n=1, ncs
				do k=1, nmodes
					inv_ov_ff_mul_F_m_fnP(n,k) = sum( inv_ov_ff(n,:)*(bigF(:,k)-f(n,k)*bigP(:)) )
				end do
			end do
			Amat = matmul( conjg(f(:,:)), TRANSPOSE(inv_ov_ff_mul_F_m_fnP) )
			b(:,:) = matmul( conjg(f(:,:)),TRANSPOSE(f(:,:)) )	
			!
			!
			!-- build alphaTensor
			do i=1, ncs
				do n=1, ncs
					do m=1, ncs
						alphaT(i,n,m)=sum(inv_ov_ff(i,:)*ov_ff(:,n)*(b(:,m)-b(:,n)) )
					end do
				end do
			end do
			!
			RHS = matmul(inv_ov_ff, ov_ff * Amat)
			do i=1, ncs
				do n=1, ncs
					packed_RHS((n-1)*ncs+i)=RHS(i,n)
				end do
			end do
			!
			do i=1, ncs
				do n=1, ncs
					do j=1, ncs
						do m=1, ncs
							mat2D((n-1)*ncs+i,(m-1)*ncs+j) =  KD(m,n)*KD(i,j) 
							mat2D((n-1)*ncs+i,(m-1)*ncs+j) = &
								mat2D((n-1)*ncs+i,(m-1)*ncs+j) + alphaT(i,n,m)*KD(j,n)
						end do
					end do
				end do
			end do
			!
			CALL solveEq_c( mat2D , packed_RHS, d_packed )
			!
			do i=1, ncs
				do n=1, ncs
					d(i,n) = d_packed((n-1)*ncs+i) 
				end do
			end do
			!
			do n=1, ncs
				do k=1, nmodes
					fdot(s,n,k) = ( inv_ov_ff_mul_F_m_fnP(n,k) - sum( d(n,:)*(f(:,k)-f(n,k)) ) )/p(n)
				end do
				!
				pdot(s,n) = sum( inv_ov_ff(n,:)* bigP(:) ) - sum( d(n,:) ) &
					+ 0.5_8*p(n)*( sum( fdot(s,n,:)*conjg(f(n,:)) &
					+ conjg(fdot(s,n,:))*f(n,:)) )
			end do

		end do

		return
	END SUBROUTINE

	SUBROUTINE update_sums_qtr( wwk, gk, y, ovm, bigL, bigW )
		real(8), intent(in)                :: wwk(:)
		real(8), intent(in)                :: gk(:)
		complex(8), intent(in)             :: y(:,:,:)
		complex(8),dimension( size(y,1),size(y,2),size(y,1),size(y,2) ),&
			intent(out)    ::  ovm
		complex(8),dimension( size(y,1),size(y,2),size(y,2) ),&
			intent(out)    ::  bigW,bigL
		integer ::  m,n,i,j


		do m=1, size(y,2)
			do n=1, size(y,2)
				do i=1, size(y,1)

					bigW(i,m,n) = dot_product( wwk(:)*y(i,m,:), y(i,n,:) )
					bigL(i,m,n) = sum( gk(:) * (conjg(y(i,m,2:)) + y(i,n,2:)) )

					do j=1, size(y,1)
						CALL calc_overlap( y(i,m,:), y(j,n,:), ovm(i,m,j,n) )
					end do

				end do
			end do
		end do

		return
	END SUBROUTINE

END MODULE



!  SUBROUTINE update_sums(sys,st)
!	 
!	 type(param),intent(in)			::  sys
!	 type(state),intent(in out)   ::  st
!	 integer								::  i,j,s
!
!	 do s=1,  sys%nq-1 
!	 	do i=1, ncs
!	 	   do j=1, ncs
!
!	 	   	 st%ov_ff(s,i,s,j) = ov(st%f(s,i,:),st%f(s,j,:))
!	 	   	 st%ov_ff(s+1,i,s,j) = ov(st%f(s+1,i,:),st%f(s,j,:))
!	 	   	 st%ov_ff(s,i,s+1,j) = ov(st%f(s,i,:),st%f(s+1,j,:))
!	 	   	 st%ov_ff(s+1,i,s+1,j) = ov(st%f(s+1,i,:),st%f(s+1,j,:))
!
!	 	     if (nmodes .ne. 1) then
!	 	   	 	st%bigW(s,i,j) = dot_product( sys%wwk , conjg(st%f(s,i,:)) * st%f(s,j,:) )
!	 	   	 	st%bigW(s+1,i,j) = dot_product( sys%wwk , conjg(st%f(s+1,i,:)) * st%f(s+1,j,:) )
!	 	   	 	st%bigL(s,i,s+1,j) = dot_product( sys%gk(s,s+1,:) , conjg(st%f(s,i,:)) + st%f(s+1,j,:) )
!	 	   	 	st%bigL(s+1,i,s,j) = dot_product( sys%gk(s+1,s,:) , conjg(st%f(s+1,i,:)) + st%f(s,j,:) )
!	 	     else if (nmodes == 1) then
!	 	   	 	st%bigW(s,i,j) = sys%wwk(1)*conjg(st%f(s,i,1)) * st%f(s,j,1)
!	 	   	 	st%bigW(s+1,i,j) = sys%wwk(1)*conjg(st%f(s+1,i,1)) * st%f(s+1,j,1)
!	 	   	 	st%bigL(s,i,s+1,j) = sys%gk(s,s+1,1)*( conjg(st%f(s,i,1)) + st%f(s+1,j,1) )
!	 	   	 	st%bigL(s+1,i,s,j) = sys%gk(s+1,s,1)*( conjg(st%f(s+1,i,1)) + st%f(s,j,1) )
!	 	     end if
!
!	 	   end do
!	 	end do
!	end do
!
!  END SUBROUTINE
!
!
!  FUNCTION ov(f1,f2)
!
!        complex(8), intent(in) :: f1( : ), f2( : )
!        complex(8)             :: ov
!        != internal variables
!        complex(8)    :: tmp1, tmp2, tmp3
!
!        != initialize
!        tmp1 = 0._8
!        tmp2 = 0._8
!        tmp3 = 0._8
!
!		  if (size(f1,1) .ne. 1) then
!			 tmp1 = dot_product(f1, f1)
!			 tmp2 = dot_product(f2, f2)
!			 tmp3 = dot_product(f1, f2)
!		  else if (size(f1,1) == 1) then
!			 tmp1 = conjg(f1(1))*f1(1)
!			 tmp2 = conjg(f2(1))*f2(1)
!			 tmp3 = conjg(f1(1))*f2(1)
!		  end if
!
!        ov = exp( -0.5_8*tmp1 - 0.5_8*tmp2 + tmp3 ) 
!
!  END FUNCTION	ov



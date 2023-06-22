import numpy as np
from scipy.integrate import complex_ode
import sys
import os

def KD( a, b ):

	if a==b:
		return 1
	else:
		return 0

def ov( f1, f2 ):

	overlap = np.exp( -0.5*np.vdot(f1,f1) - 0.5*np.vdot(f2,f2) + np.vdot(f1,f2) )
	return overlap

class state(object):

	def __init__( self, *args ):

		if len(args) == 2:
			prms = args[0]
			ncs = args[1]

			self.ncs = ncs
			self.t = 0
			self.p = prms.p0*np.ones( (prms.nl, ncs), dtype='complex128' )
			self.pdot = np.zeros( (prms.nl, ncs), dtype='complex128' )
			self.one_ph_f = 0.1

			self.y = np.zeros( (prms.nl, ncs, prms.nmodes), dtype='complex128' )
			self.ydot = np.zeros( (prms.nl, ncs, prms.nmodes), dtype='complex128' )


			if prms.cav_ini == 0:
				for n in range(self.ncs):
					self.y[:,n,:] = prms.offset*n

				for l in prms.qb_ini:
					self.p[l,0] = 1.0

			if prms.cav_ini == -1:
				for n in range(self.ncs):
					self.y[:,n,:] = prms.offset*n

				for l in prms.qb_ini:
					self.y[l,:,:] = 0.0
					self.p[l,0] = 1/(2*self.one_ph_f)
					self.p[l,1] = - 1/(2*self.one_ph_f)
					for mu in range(prms.nmodes):
						self.y[l,0,mu] = self.one_ph_f*( prms.O[0,mu] )
						self.y[l,1,mu] = - self.one_ph_f*( prms.O[0,mu] )

			elif prms.cav_ini > 0:

				for l in range(prms.nl):
					for n in range(self.ncs):
						self.y[l,n,:] = prms.offset*n + prms.cav_ini*( prms.O[0,:] )

				for l in prms.qb_ini:
					self.p[l,0] = 1.0

			#elif prms.cav_ini > 0:

			#	for l in range(prms.nl):
			#		for n in range(self.ncs):
			#			self.y[l,n,:] = prms.offset*n

			#	for l in prms.qb_ini:
			#		self.p[l,0] = 1.0
			#		self.y[l,0,:] = prms.cav_ini*( prms.O[0,:] )

			self.ovm = np.zeros( (prms.nl, ncs,  prms.nl, ncs), dtype='complex128' )
			self.bigW = np.zeros( (prms.nl, ncs, ncs), dtype='complex128' )
			self.bigU = np.zeros( (prms.nl, ncs, ncs), dtype='complex128' )
			self.bigL = np.zeros( (prms.nl, ncs,prms.nl, ncs), dtype='complex128' )
			self.update_sums( prms )
			self.normalize()

		elif len(args) == 3:

			prms = args[0]
			self.t = args[1]
			label = args[2]

			folder_name = prms.save_dir+'/'+prms.param_char()
			array = np.loadtxt( folder_name+'/p_'+prms.param_char()+'_'+label+'.d',\
					dtype=np.complex_ )
			self.ncs = len(array) // prms.nl
			self.p = array.reshape( (prms.nl, self.ncs) )
			array = np.loadtxt( folder_name+'/y_'+prms.param_char()+'_'+label+'.d',\
					dtype=np.complex_)
			self.y = array.reshape( (prms.nl, self.ncs, prms.nmodes) ) 

			self.pdot = np.zeros( (prms.nl, self.ncs), dtype='complex128' )
			self.ydot = np.zeros( (prms.nl, self.ncs, prms.nmodes), dtype='complex128' )

			self.ovm=np.zeros( (prms.nl, self.ncs,  prms.nl, self.ncs), dtype='complex128' )
			self.bigW = np.zeros( (prms.nl, self.ncs, self.ncs), dtype='complex128' )
			self.bigU = np.zeros( (prms.nl, self.ncs, self.ncs), dtype='complex128' )
			self.bigL = np.zeros( (prms.nl, self.ncs,prms.nl, self.ncs), dtype='complex128' )
			self.update_sums( prms )
			self.normalize()

	def save_state( self, prms, label ):

		folder_name = prms.save_dir+'/'+prms.param_char()
		if not os.path.isdir(folder_name):
			os.mkdir( folder_name )
		np.savetxt( folder_name+'/p_'+prms.param_char()+'_'+label+'.d', self.p.reshape(-1) )
		np.savetxt( folder_name+'/y_'+prms.param_char()+'_'+label+'.d', self.y.reshape(-1) )

	def update_sums( self, prms ):

		for m in range( self.ncs ):
			for n in range( self.ncs ):
				for i in range( prms.nl ):

					self.bigW[i,m,n] = \
							np.sum( prms.Wk[:]*np.conj(self.y[i,m,:])*self.y[i,n,:] )
					self.bigU[i,m,n] = \
							np.sum( prms.O[0,:] * ( np.conj(self.y[i,m,:]) + self.y[i,n,:] ) )


					for j in range( prms.nl ):
						self.ovm[i,m,j,n] = ov( self.y[i,m,:], self.y[j,n,:] )
						self.bigL[i,m,j,n] = np.sum( \
								prms.gk[i,j,:] * (np.conj(self.y[i,m,:]) + self.y[j,n,:]) )

	def norm( self ):

		tmp = 0
		for i in range( len( self.p[:,0] ) ):
			for m in range( self.ncs ):
				for n in range( self.ncs ):
					tmp += np.conj(self.p[i,m])*self.p[i,n]*self.ovm[ i,m,i,n ]

		if np.imag(tmp) > 1e-8:
			print('ERROR in norm: imaginary of Y is non-zero')


		return np.sqrt(np.real(tmp))

	def normalize( self ):

		self.p /= self.norm()

	def lvl_occupation( self, s ):

		tmp = 0.0
		for m in range( self.ncs ):
			for n in range( self.ncs ):
				tmp += np.conj(self.p[s,m])*self.p[s,n]*self.ovm[ s,m,s,n ]

		if np.imag(tmp) > 1e-8:
			print('ERROR in norm: imaginary of Y is non-zero')

		return np.real(tmp)

	def tot_ph_nb( self ):

		p = self.p
		pc = np.conj( p )
		y = self.y
		yc = np.conj( y )

		tmp = 0
		nl = len( p[:,0] )
		for i in range( nl ):
			for m in range( self.ncs ):
				for n in range( self.ncs ):
					tmp += pc[i,m]*p[i,n]*np.sum(yc[i,m,:]*y[i,n,:])*self.ovm[ i,m,i,n ]

		if np.imag(tmp) > 1e-8:
			print('ERROR in norm: imaginary of Y is non-zero')

		return np.real(tmp)

	def y_OB( self, prms, sig, rotating_frame=False ):

		y_out = np.zeros( (prms.nl, self.ncs), dtype='complex128')

		for i in range( prms.nl ):
			for n in range( self.ncs ):

				y_out[i,n] = np.sum( prms.O[sig,:]*self.y[i,n,:] )

		if rotating_frame:
			y_out *= np.exp( 1j*prms.wd*self.t ) 

		return y_out

	def y_OB_lns( self, prms, lvl, n, sig, rotating_frame=False ):

		return np.sum( prms.O[sig,:]*self.y[lvl,n,:] )

	def X( self, prms, l, s ):

		p = self.p
		y = self.y
		pc = np.conj(p)
		yc = np.conj(y)
		tmp = 0
		for n  in range( self.ncs ):
			for m  in range( self.ncs ):
				tmp += pc[l,m]*p[l,n]*self.ovm[l,m,l,n]*np.sum( ( yc[l,m,:] + y[l,n,:] )*prms.O[s,:] )

		return tmp

	def X_full( self, prms, s ):

		p = self.p
		y = self.y
		pc = np.conj(p)
		yc = np.conj(y)
		tmp = 0
		for l in range(prms.nl):
			for n  in range( self.ncs ):
				for m  in range( self.ncs ):
					tmp += pc[l,m]*p[l,n]*self.ovm[l,m,l,n]*np.sum( ( yc[l,m,:] + y[l,n,:] )*prms.O[s,:] )

		return tmp

	def P( self, prms, l, s ):

		p = self.p
		y = self.y
		pc = np.conj(p)
		yc = np.conj(y)
		tmp = 0
		for n  in range( self.ncs ):
			for m  in range( self.ncs ):
				tmp += 1j*pc[l,m]*p[l,n]*self.ovm[l,m,l,n]*np.sum( ( yc[l,m,:] - y[l,n,:] )*prms.O[s,:] )

		return tmp

	def P_full( self, prms, s ):

		p = self.p
		y = self.y
		pc = np.conj(p)
		yc = np.conj(y)
		tmp = 0
		for l in range(prms.nl):
			for n  in range( self.ncs ):
				for m  in range( self.ncs ):
					tmp += 1j*pc[l,m]*p[l,n]*self.ovm[l,m,l,n]*np.sum( ( yc[l,m,:] - y[l,n,:] )*prms.O[s,:] )

		return tmp

	def sig_mode_ph_nb( self, prms, sig ):

		p = self.p
		pc = np.conj( p )
		y = self.y
		yc = np.conj( y )

		tmp = 0
		ncs = len( p[0,:] )
		nl = len( p[:,0] )
		for i in range( nl ):

			y_OB = np.zeros( ncs, dtype='complex128' )
			for n in range( ncs ):
				y_OB[n] = np.sum( prms.O[sig,:] * y[i,n,:] )
			yc_OB = np.conj( y_OB )

			for m in range( self.ncs ):
				for n in range( self.ncs ):
					tmp += pc[i,m]*p[i,n]*np.sum(yc_OB[m]*y_OB[n])*self.ovm[ i,m,i,n ]

		if np.imag(tmp) > 1e-8:
			print('ERROR in norm: imaginary of Y is non-zero')

		return np.real(tmp)

	def n_up_x( self, prms):


		fnx = f_nx( self, prms )
		out = np.zeros( (imax-imin), dtype='float64' )

		for k in range( prms.nmodes ):
			n_k = 0.0
			for l in range( prms.nl ):
				for n in range( self.ncs ):
					for m in range( self.ncs ):
						tmp = np.conj( fnx[l,n,k] ) * fnx[l,m,k] 
						n_k += np.conj(self.p[n])*self.p[m]*tmp*self.ovm[l,n,l,m]

			out[k] = n_k

		return out

	def energy( self, prms ):

		p = self.p
		pc = np.conj(p)
		y = self.y
		yc = np.conj(y)
		ovm = self.ovm

		tmp = 0
		for i in range( prms.nl ):
			for m in range( self.ncs ):
				for n in range( self.ncs ):

					tmp +=  pc[i,m]*p[i,n]*ovm[i,m,i,n]\
							*( prms.w_qb[i] + self.bigW[i,m,n] \
							+ prms.A_d*np.cos(prms.wd*self.t)*self.bigU[i,m,n] )

					if (i<prms.nl-1):
						tmp += pc[i,m]*p[i+1,n]*ovm[i,m,i+1,n]*self.bigL[i,m,i+1,n] \
								+pc[i+1,m]*p[i,n]*ovm[i+1,m,i,n]*self.bigL[i+1,m,i,n]

		if np.imag(tmp)>1e-5:
			print('EROOR: energy imaginary part non-zero')


		return tmp 

	def dE_dpc_sj( self, prms, s ):

		'''
		Outputs the deprivative of the energy by pj*.
		Output: vector of length ncs
		'''

		p = self.p
		pc = np.conj(p)
		y = self.y
		yc = np.conj(y)

		out = np.zeros( self.ncs, dtype='complex128' )

		for j in range( self.ncs ):
			out[j] = np.sum( p[s,:]*self.ovm[s,j,s,:]*( prms.w_qb[s] \
					+ self.bigW[s,j,:] \
					+ prms.A_d*np.cos( prms.wd*self.t ) * self.bigU[s,j,:]   ))
			if s<prms.nl-1:
				out[j] += np.sum( p[s+1,:]*self.ovm[s,j,s+1,:]*self.bigL[s,j,s+1,:] )
			if s>0:
				out[j] += np.sum( p[s-1,:]*self.ovm[s,j,s-1,:]*self.bigL[s,j,s-1,:]) 

		#if np.imag(tmp) > 1e-8:
		#    print('EROOR: P_j imaginary part non-zero')

		return out

	def dE_dyc_sj( self, prms, s, j, k ):

		p = self.p
		pc = np.conj(p)
		y = self.y
		yc = np.conj(y)

		tmp=0
		for n in range(self.ncs):
			tmp += pc[s,j]*p[s,n]*self.ovm[s,j,s,n]*( prms.Wk[k]*y[s,n,k] \
					+ prms.A_d*np.cos( prms.wd*self.t )*prms.O[0,k] \
					+ (y[s,n,k]-0.5*y[s,j,k])*( prms.w_qb[s] + self.bigW[s,j,n]\
					+ prms.A_d*np.cos(prms.wd*self.t)*self.bigU[s,j,n] ) ) \
					-0.5*pc[s,n]*p[s,j]*self.ovm[s,n,s,j]*y[s,j,k]*( prms.w_qb[s] \
					+ self.bigW[s,n,j] \
					+ prms.A_d*np.cos(prms.wd*self.t)*self.bigU[s,n,j] )

			if s<prms.nl-1:
				tmp += ( pc[s,j]*p[s+1,n]*self.ovm[s,j,s+1,n]*( prms.gk[s,s+1,k]+(y[s+1,n,k]-0.5*y[s,j,k])*self.bigL[s,j,s+1,n] )\
						-0.5*pc[s+1,n]*p[s,j]*self.ovm[s+1,n,s,j]*y[s,j,k]*self.bigL[s+1,n,s,j])

			if s>0:
				tmp += ( pc[s,j]*p[s-1,n]*self.ovm[s,j,s-1,n]*( prms.gk[s,s-1,k]+(y[s-1,n,k]-0.5*y[s,j,k])*self.bigL[s,j,s-1,n] ) \
						-0.5*pc[s-1,n]*p[s,j]*self.ovm[s-1,n,s,j]*y[s,j,k]*self.bigL[s-1,n,s,j] )


		#if np.imag(tmp) > 1e-8:
		#    print('EROOR: P_j imaginary part non-zero')

		return tmp

	def calc_derivatives( self, prms ):

		for s in range( prms.nl ):

			p = self.p[s,:]
			pc = np.conj(p)
			y = self.y[s,:,:]
			yc = np.conj(y)

			dE_dyc = np.zeros( (self.ncs,prms.nmodes), dtype='complex128')
			P = np.zeros( self.ncs, dtype='complex128' )
			Y = np.zeros( (self.ncs,prms.nmodes), dtype='complex128' )
			b = np.zeros( (self.ncs,self.ncs), dtype='complex128' )
			ovm = np.zeros( (self.ncs, self.ncs), dtype='complex128' )
			ovm_inv = np.zeros( (self.ncs, self.ncs), dtype='complex128' )
			ovm_inv_mul_Y_m_ynP = np.zeros( (self.ncs, prms.nmodes), dtype='complex128' )
			alpha = np.zeros( (self.ncs, self.ncs, self.ncs), dtype='complex128' )
			RHS = np.zeros( (self.ncs, self.ncs), dtype='complex128' )
			packed_RHS = np.zeros( self.ncs**2, dtype='complex128' )
			mat = np.zeros( (self.ncs, self.ncs, self.ncs, self.ncs), dtype='complex128' )
			mat2D = np.zeros( (self.ncs**2, self.ncs**2), dtype='complex128' )
			a = np.zeros( (self.ncs, self.ncs), dtype='complex128' )
			d = np.zeros( (self.ncs, self.ncs), dtype='complex128' )

			ovm = self.ovm[s,:,s,:]
			ovm_inv = np.linalg.inv(ovm)

			dE_dpc = self.dE_dpc_sj(prms,s)
			for n in range(self.ncs):
				for k in range(prms.nmodes):
					dE_dyc[n,k] = self.dE_dyc_sj(prms,s,n,k)

			P = -1j*dE_dpc
			for k in range(prms.nmodes):
				Y[:,k] = -1j*( dE_dyc[:,k]/pc+0.5*( dE_dpc + np.conj(dE_dpc)*p/pc )*y[:,k] )

			for n in range(self.ncs):
				for k in range(prms.nmodes):
					ovm_inv_mul_Y_m_ynP[n,k] = np.sum( ovm_inv[n,:]*(Y[:,k]-y[n,k]*P[:]) )


			A = np.matmul( yc[:,:], ovm_inv_mul_Y_m_ynP.transpose() )

			#-- build b matrice
			b[:,:] = np.dot( yc[:,:],y[:,:].transpose() )

			#-- build the alpha matrix
			for i in range(self.ncs):
				for n in range(self.ncs):
					for m in range(self.ncs):
						alpha[i,n,m] = np.sum( ovm_inv[i,:]*ovm[:,n]*(b[:,m]-b[:,n]) )

			ovm_t_A = ovm * A
			RHS = np.matmul(ovm_inv, ovm_t_A)
			ncs = self.ncs
			for i in range(self.ncs):
				for n in range(self.ncs):
					packed_RHS[n*ncs+i]=RHS[i,n]

			for i in range( self.ncs ):
				for n in range( self.ncs ):
					for j in range( self.ncs ):
						for m in range( self.ncs ):
							mat2D[n*ncs+i,m*ncs+j] = KD(m,n)*KD(i,j) + alpha[i,n,m]*KD(j,n)


			packed_sol = np.linalg.solve( mat2D,packed_RHS )
			for i in range(self.ncs):
				for n in range(self.ncs):
					d[i,n] = packed_sol[n*ncs+i]


			for n in range(self.ncs):
				for k in range(prms.nmodes):
					self.ydot[s,n,k] = ( ovm_inv_mul_Y_m_ynP[n,k] - np.sum( d[n,:]*(y[:,k]-y[n,k]) ) )/p[n]

				self.pdot[s,n] = np.matmul( ovm_inv[n,:], P[:] ) - np.sum( d[n,:] ) \
						+0.5*p[n]*( np.sum( self.ydot[s,n,:]*yc[n,:] \
						+ np.conj(self.ydot[s,n,:])*y[n,:]) )
						#+ np.conj(self.zdot[s,n,:])*z[n,:] ) )
									   #+ np.conj(self.zdot[s,n,:])*z[n,:] ) )



   #dRHS_f=matmultiply_c(CONJG(st%f),TRANSPOSE(rhsA_f))
   #dRHS_f=st%ov_ff*dRHS_f
   #dRHS_f=matmultiply_c(inv_ov_ff,dRHS_f)
   #dRHS_h=matmultiply_c(CONJG(st%h),TRANSPOSE(rhsA_h))
   #dRHS_h=st%ov_hh*dRHS_
   #dRHS_h=matmultiply_c(inv_ov_hh,dRHS_h)

   #do i=1, st%ncs
   #  do n=1, st%ncs
   #    packed_dRHS_f((n-1)*st%ncs+i)=dRHS_f(i,n)
   #    packed_dRHS_h((n-1)*st%ncs+i)=dRHS_h(i,n)
   #  end do
   #end do


   #!-- build alphaTensor

   #! -- test area
   # ! if ((st%t > sys%tmax/5) .OR. (st%ncs == sys%ncsadd) ) then
   # !  call clearProgressBar
   # !  print*, 'begin superInverse number :  ', counter
   # !  call CPU_time(startTime)
   # !  call superInverse_f(st%ncs, alphaT_f, packedSol_f, packed_dRHS_f)
   # !  call CPU_time(endTime)
   # !  st%savePackedSol=packedSol_f
   # !  print*, 'ended. CPUtime : ', endTime-startTime, 's'
   # !  print*, 'begin direct solve ... '
   # !  call CPU_time(startTime)
   # !  CALL directInverse(st%ncs, alphaT_f, packedSol_f, packed_dRHS_f)
   # !  call CPU_time(endTime)
   # !  print*, 'ended. CPUtime : ', endTime-startTime, 's'
   # !  print*, 'error : ', NORM2(ABS(packedSol_f-st%savePackedSol))/NORM2(ABS(packedSol_f))
   # !  if (counter >= 100) stop "100 runs done. end job."
   # !  counter=counter+1
   # !  call progress_bar(st%t, sys%tmax)
   # ! end if

   #! -- SuperInverse detection
   #if ((superInverseFlag .eqv. .false.) &
   #     .OR. (st%ncs < 12) &
   #     .OR. (sys%fastCalcDeriv==1)) then
   #  CALL directInverse(st%ncs, alphaT_f, packedSol_f, packed_dRHS_f)
   #  CALL directInverse(st%ncs, alphaT_h, packedSol_h, packed_dRHS_h)
   #else
   #  ! call clearProgressBar
   #  ! print*, 'begin superinverse...'
   #  ! call CPU_time(startTime)
   #  CALL superInverse_f(st%ncs, alphaT_f, packedSol_f, packed_dRHS_f)
   #  ! call CPU_time(endTime)
   #  ! print*, 'ended. CPUtime : ', endTime-startTime, 's'
   #  CALL superInverse_h(st%ncs, alphaT_h, packedSol_h, packed_dRHS_h)
   #end if

   #!-- system uncsack
   #! print*, 'uncsack'
   #do i=1, st%ncs
   #  do n=1, st%ncs
   #    a_f(n,i)=packedSol_f((n-1)*st%ncs+i)
   #    a_h(n,i)=packedSol_h((n-1)*st%ncs+i)
   #  end do
   #end do

   #a_f=matmultiply_c(st%ov_ff, a_f)
   #a_f=TRANSPOSE(a_f/st%ov_ff)  !! -- This TRANSPOSE is a mystery.

   #a_h=matmultiply_c(st%ov_hh, a_h)
   #a_h=TRANSPOSE(a_h/st%ov_hh)  !! -- This TRANSPOSE is a mystery.

#  !-- fDot and hdot extraction
#  ! print*, 'fDot and hdot'
#
#  tempMatrix_f=matmultiply_c(inv_ov_ff,st%ov_ff*TRANSPOSE(a_f))
#  do i=1, st%ncs
#    tempfDot(i,:)=SUM(tempMatrix_f(i,:))*st%f(i,:)
#  end do
#  st%fDot= rhsA_f-matmultiply_c(tempMatrix_f,st%f)+tempfDot
#  do i=1, st%ncs
#    st%fDot(i,:)=st%fDot(i,:)/st%p(i)
#  end do
#
#
#  tempMatrix_h=matmultiply_c(inv_ov_hh,st%ov_hh*TRANSPOSE(a_h))
#  do i=1, st%ncs
#    temphDot(i,:)=SUM(tempMatrix_h(i,:))*st%h(i,:)
#  end do
#  st%hDot= rhsA_h-matmultiply_c(tempMatrix_h,st%h)+temphDot
#  do i=1, st%ncs
#    st%hDot(i,:)=st%hDot(i,:)/st%q(i)
#  end do
#
#  !-- evaluate pDot and qDot
#  ! print*, 'pdot and qdot'
#  tempMatrixTer_f= MATMUL(inv_ov_ff,bigP)
#  st%pDot= 0.5_rl*( (/ (a_f(n,n), n=1, st%ncs) /) + st%p*(/ (conjg(a_f(m,m)), m=1, st%ncs) /) /CONJG(st%p) )
#  st%pdot=st%pDot + tempMatrixTer_f
#  st%pDot=st%pDot - SUM(tempMatrix_f, dim=2)
#
#  tempMatrixTer_h= MATMUL(inv_ov_hh,bigQ)
#  st%qDot= 0.5_rl*( (/ (a_h(m,m), m=1, st%ncs) /) + st%q*(/ (conjg(a_h(m,m)), m=1, st%ncs) /) /CONJG(st%q) )
#  st%qdot=st%qDot + tempMatrixTer_h
#  st%qDot=st%qDot - SUM(tempMatrix_h, dim=2)
#
#		:q




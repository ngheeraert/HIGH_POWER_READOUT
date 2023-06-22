import numpy as np
from scipy.integrate import complex_ode
from .fortran_module import fortran_module
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


			if prms.cav_ini > -1e-7:

				for l in range(prms.nl):
					self.y[l,0,:] = prms.cav_ini * ( prms.O[0,:] )

				if self.ncs > 1:

					phi_offset = 2*np.pi/(self.ncs-1)
					for n in range(1,self.ncs):

						for l in range(prms.nl):
							self.y[l,n,:] = ( prms.cav_ini + prms.offset*np.exp( 1j*phi_offset*n ) )*prms.O[0,:] 

				for l in prms.qb_ini:
					self.p[l,0] = 1.0

			elif prms.cav_ini == -1:
				for n in range(self.ncs):
					self.y[:,n,:] = prms.offset*n

				for l in prms.qb_ini:
					self.y[l,:,:] = 0.0
					self.p[l,0] = 1/(2*self.one_ph_f)
					self.p[l,1] = - 1/(2*self.one_ph_f)
					for mu in range(prms.nmodes):
						self.y[l,0,mu] = self.one_ph_f*( prms.O[0,mu] )
						self.y[l,1,mu] = - self.one_ph_f*( prms.O[0,mu] )

			else:
				print('ERROR: wrong value of cav_ini')



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

					for l in range( prms.nl ):

						tmp += pc[l,m]*p[i,n]*ovm[l,m,i,n]*self.bigL[l,m,i,n]

		if np.imag(tmp)>1e-5:
			print('EROOR: energy imaginary part non-zero')


		return tmp 

	def calculate_derivatives( self, prms ):

		self.pdot, self.ydot = fortran_module.calc_derivatives( prms.w_qb, prms.wd, prms.A_d, self.t, prms.Wk, prms.gk, self.ovm, self.bigW, self.bigL, self.bigU, prms.O, self.p, self.y )

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
					tmp += pc[i,m]*p[i,n]*yc_OB[m]*y_OB[n]*self.ovm[ i,m,i,n ]

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



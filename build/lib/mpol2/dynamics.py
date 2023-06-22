import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from .state import state
from .system_fortran import system_fortran

class dynamics(object):

	def __init__( self, *args ):

		prms = args[0]
		if len(args) == 1:
			if prms.device == 'TRSM':
				self.st = state( prms, prms.ncs_ini )
			elif prms.device == 'QUTR':
				self.st = state_quantromon( prms, prms.ncs_ini )

		elif len(args) > 1:
			self.st = args[1]

		self.ost = deepcopy( self.st )
		self.oost = deepcopy( self.st )

		self.times = []
		self.st_hist = []
		self.error_hist = []
		self.adding_times = []

		#folder_name = prms.save_dir+'/'+prms.param_char()
		#if  os.path.isdir(folder_name):
		#    print('-- SIMULATION AVAILABLE: load simulation')
		#    sys.exit()

	def truncate_evolution( self, time ):


		percentage = (self.st.t-time)/self.st.t
		steps = len( self.times )        

		for i in range( int( steps*percentage) ):
			self.times.pop()
			self.st_hist.pop()
			if self.error_hist:
				self.error_hist.pop()

		self.st = self.st_hist[-1]
		self.ost = self.st_hist[-2]
		self.oost = self.st_hist[-3]

	def time_evolve( self, prms, tmax, print_delay ):

		t_last_print = -1e5

		adding_delay = 0.2
		time_added = 0
		max_cs = 2
		t_ref=0

		t_error = 0
		count = 0
		while ( self.st.t < tmax ):

			if ( self.st.t - t_last_print > print_delay ):

				self.times.append( self.st.t )
				self.st_hist.append( deepcopy(self.st) )
				t_last_print = self.st.t

		#		if prms.calc_error:
		#			if self.st.t - t_ref < prms.dt:
		#				if len(self.error_hist) == 0:
		#					self.error_hist.append( 0 )   
		#				else:
		#					self.error_hist.append( self.error_hist[-1] )   
		#			else:
		#				t1 = time.time()
		#				for i in range(4):
		#					self.evolve_RK4_fortran(  prms, prms.dt/100 )
		#				self.error_hist.append( np.real(self.error(prms)) )

		#			if ( prms.error_thr < self.error_hist[-1] \
				#					and self.st.ncs < prms.ncs_max ):
		#				self.add_coherent_state(prms,'like_heaviest')
		#				t_ref = self.st.t

				#for i in range(3):
				#    dt = prms.dt*1e-3
				#    ost, oost = evolve_RK4( prms, st, ost, oost, dt )

			#if adding_coherent_states == True:
			#    if (st.t-time_added>adding_delay) and (st.ncs < max_cs):
			#        st = add_coherent_state(prms, st)
			#        time_added = st.t
			#        t_ref = time_added

			if np.abs(self.st.t-t_ref)<5*prms.dt/1000:
				dt = prms.dt/10000
			elif np.abs(self.st.t-t_ref)<5*prms.dt/100:
				dt = prms.dt/1000
			elif np.abs(self.st.t-t_ref)<5*prms.dt/10:
				dt = prms.dt/100
			elif np.abs(self.st.t-t_ref)<5*prms.dt:
				dt = prms.dt/10
			else:
				dt = prms.dt

			self.evolve_RK4_fortran( prms, dt )

			if self.st.ncs == self.oost.ncs:
				delta_norm = np.abs( self.st.norm() - self.oost.norm() )
				if  np.abs( delta_norm ) > prms.norm_errmax:
					if prms.verbose:
						print('-- FIXING: delta_norm =', delta_norm, 't=',self.st.t )
					self.correct_time_step_norm( prms, dt, delta_norm )

	def time_evolve_2( self, prms, tmax, print_delay ):

		print_times = list( np.linspace( self.st.t, tmax + 10*prms.dt, 10 ) )

		tref = 0.0
		while ( self.st.t < tmax ):

			if prms.time_bar:
				if print_times[0] < self.st.t:
					print_times.pop(0)
					print( '.', end='' )

			self.times.append( self.st.t )
			self.st_hist.append( deepcopy(self.st) )

			if prms.calc_error:

				if self.st.t - tref <= 0.1:
					if len(self.error_hist) == 0:
						self.error_hist.append( 0 )   
					else:
						self.error_hist.append( self.error_hist[-1] )   

				else:
					for i in range(3):
						self.evolve_RK4_fortran(  prms, prms.dt/1000 )
					self.error_hist.append( np.real(self.error(prms)) )

			#	if ( prms.error_thr < self.error_hist[-1] \
			#					and self.st.ncs < prms.ncs_max ):
			#		self.add_coherent_state(prms,'like_heaviest')
			#		t_ref = self.st.t

			if self.st.ncs < prms.ncs_max:

				self.all_fortran_time_evolve( prms, min( self.st.t + print_delay,\
						prms.t_add_arr[ self.st.ncs-1 ] ) )

				if self.st.t > prms.t_add_arr[ self.st.ncs-1 ]:
					self.add_coherent_state( prms, '0' )
					tref = self.st.t

			else:

				self.all_fortran_time_evolve( prms, min( self.st.t + print_delay, tmax + 0.5*prms.dt ) )

	def all_fortran_time_evolve( self, prms, tf ):

		if prms.device == 'TRSM': 
			new_p, new_y, new_t = system_fortran.time_evolve( prms.err_lim, prms.w_qb, prms.wd, prms.A_d, self.st.t, tf, prms.dt, prms.Wk, prms.gk, self.st.ovm, self.st.bigW, self.st.bigL, self.st.bigU, prms.O, self.st.p, self.st.y )
		elif prms.device == 'QUTR': 
			new_p, new_y, new_t = system_fortran.time_evolve_qtr( prms.w_qb, prms.wd, prms.A_d, self.st.t, tf, prms.dt, prms.Wk, prms.g_qc, prms.gk, self.st.ovm, self.st.bigW, self.st.bigL, self.st.p, self.st.y )

		self.st.p = new_p
		self.st.y = new_y
		self.st.t = new_t

		if prms.device == 'TRSM': 
			self.st.ovm, self.st.bigL, self.st.bigW, self.st.bigU \
					= system_fortran.update_sums( prms.Wk, prms.gk, prms.O, self.st.y )
		elif prms.device == 'QUTR': 
			self.st.ovm, self.st.bigL, self.st.bigW \
					= system_fortran.update_sums_qtr( prms.Wk, prms.gk, self.st.y )

	#-- adaptive RK4 testing routines
	def time_evolve_2_TEST( self, prms, tmax, print_delay ):

		error_arr = np.zeros( int( (tmax/prms.dt)*200 ), dtype='float64' )
		t_ref=0
		while ( self.st.t < tmax ):

			self.times.append( self.st.t )
			#print( len(self.st_hist) )
			self.st_hist.append( deepcopy(self.st) )

			if prms.calc_error:
				if self.st.t - t_ref < 10*prms.dt:
					if len(self.error_hist) == 0:
						self.error_hist.append( 0 )   
					else:
						self.error_hist.append( self.error_hist[-1] )   
				else:
					t1 = time.time()
					for i in range(4):
						self.evolve_RK4_fortran(  prms, prms.dt/100 )
					self.error_hist.append( np.real(self.error(prms)) )

			new_error_arr, time_arr, slow_factor_arr = self.all_fortran_time_evolve( prms, min( self.st.t + print_delay, tmax + 0.5*prms.dt ),error_arr )

		print(slow_factor_arr)
		plt.plot( time_arr,  new_error_arr )
		plt.xlim(0, tmax)
		plt.show()
		plt.plot( time_arr,  slow_factor_arr )
		plt.xlim(0.00, tmax)
		plt.ylim(-2, 20)
		plt.show()
		plt.plot( time_arr,  slow_factor_arr )
		plt.xlim(0.003, 0.0035)
		plt.show()
		plt.plot( time_arr,  slow_factor_arr )
		plt.xlim(0.003, 0.0031)
		plt.show()

	def all_fortran_time_evolve_TEST( self, prms, tf, error_arr ):

		if prms.device == 'TRSM': 
			new_p, new_y, new_t, new_error_arr, time_arr, slow_factor_arr = system_fortran.time_evolve( error_arr, 0.00001, prms.w_qb, prms.wd, prms.A_d, self.st.t, tf, prms.dt, prms.Wk, prms.gk, self.st.ovm, self.st.bigW, self.st.bigL, self.st.bigU, prms.O, self.st.p, self.st.y )
		elif prms.device == 'QUTR': 
			new_p, new_y, new_t = system_fortran.time_evolve_qtr( prms.w_qb, prms.wd, prms.A_d, self.st.t, tf, prms.dt, prms.Wk, prms.g_qc, prms.gk, self.st.ovm, self.st.bigW, self.st.bigL, self.st.p, self.st.y )

		self.st.p = new_p
		self.st.y = new_y
		self.st.t = new_t

		if prms.device == 'TRSM': 
			self.st.ovm, self.st.bigL, self.st.bigW, self.st.bigU \
					= system_fortran.update_sums( prms.Wk, prms.gk, prms.O, self.st.y )
		elif prms.device == 'QUTR': 
			self.st.ovm, self.st.bigL, self.st.bigW \
					= system_fortran.update_sums_qtr( prms.Wk, prms.gk, self.st.y )

		return new_error_arr, time_arr, slow_factor_arr

	def correct_time_step_norm( self, prms, dt, delta_norm_in ):

		st_tofix = deepcopy( self.st )
		oost_tofix = deepcopy( self.oost )
		delta_norm = delta_norm_in
		slowfactorFix = 1.
		fixtry = 1
		slowfactor = 0.245

		while ( (np.abs(delta_norm) > prms.norm_errmax) and (fixtry < 20) ):

			if prms.verbose or fixtry>13:
				print(fixtry, end=' ')
			fixtry += 1
			slowfactor *= 2        #-- keep track of the number of tries
			self.st = deepcopy( oost_tofix )		#-- rewind evolution by 2 steps

			#-- RE_EVOLVE

			while (st_tofix.t+dt > self.st.t):
				self.evolve_RK4_fortran( prms, dt/slowfactor )

			#-- recalculate the energy error
			delta_norm = np.abs( self.st.norm() - oost_tofix.norm() )


		if np.abs(delta_norm) < prms.norm_errmax :

			#print( "No fix for ",(cnt)," steps" )
			if prms.verbose:
				print("SUCCESSFUL, fix attempts:"\
						,(fixtry-1)," time=",np.round(self.st.t,5) ," New_delta_norm=",np.round(delta_norm,8) )

		else:

			print("==================================")
			print("ABORT -- error in the norm is too high")
			print("==================================")
			sys.exit()

	def add_coherent_state( self, prms, displacement ):

		self.adding_times.append( self.st.t  )

		new_ncs = self.st.ncs+1 
		new_st = state( prms, new_ncs )

		new_st.p[:,0:new_ncs-1] = self.st.p
		new_st.p[:,new_ncs-1] = prms.p0
		new_st.y[:,0:new_ncs-1,:] = self.st.y

		if displacement == '0':
			new_st.y[:,new_ncs-1,:] = 0.0
		elif displacement == 'like_heaviest':
			for s in range( prms.nl ):

				n_heaviest = np.argmax( np.abs(self.st.p[s,:])**2 )
				new_st.y[s,new_ncs-1,:] = 1.1*new_st.y[s,n_heaviest,:]

		new_st.t = self.st.t

		new_st.update_sums( prms )
		new_st.normalize()

		self.st = new_st

		print('====================')
		print( "CS ADDED: from ",self.st.ncs-1," to ",self.st.ncs, "at t=", self.st.t )
		print('====================')

	def evolve_RK4_fortran( self, prms, dt ):

		self.oost = deepcopy( self.ost )
		self.ost = deepcopy( self.st )
		midst = deepcopy( self.st )

		if prms.device == 'TRSM':

			self.st.pdot, self.st.ydot= system_fortran.calc_derivatives_2( prms.w_qb, prms.wd, prms.A_d, self.st.t, prms.Wk, prms.gk, self.st.ovm, self.st.bigW, self.st.bigL, self.st.bigU, prms.O, self.st.p, self.st.y )

			self.st.y = self.st.y + self.st.ydot*dt/6.0
			self.st.p = self.st.p + self.st.pdot*dt/6.0
			self.st.t += dt/6.0

			#== SECOND STEP OF RK4 ==========
			midst.y = self.ost.y + 0.5*dt*self.st.ydot
			midst.p = self.ost.p + 0.5*dt*self.st.pdot

			midst.ovm, midst.bigL, midst.bigW, midst.bigU \
					= system_fortran.update_sums( prms.Wk, prms.gk, prms.O, midst.y )

			midst.pdot, midst.ydot = system_fortran.calc_derivatives_2( prms.w_qb, prms.wd, prms.A_d, midst.t, prms.Wk, prms.gk, midst.ovm, midst.bigW, midst.bigL, midst.bigU, prms.O, midst.p, midst.y )

			self.st.y = self.st.y + midst.ydot*dt/3.0
			self.st.p = self.st.p + midst.pdot*dt/3.0
			self.st.t += dt/3.0

			#== THIRD STEP OF RK4 ==========
			midst.y = self.ost.y + 0.5*dt*midst.ydot
			midst.p = self.ost.p + 0.5*dt*midst.pdot
			midst.ovm, midst.bigL, midst.bigW, midst.bigU \
					= system_fortran.update_sums( prms.Wk, prms.gk, prms.O, midst.y )

			midst.pdot, midst.ydot = system_fortran.calc_derivatives_2( prms.w_qb, prms.wd, prms.A_d, midst.t, prms.Wk, prms.gk, midst.ovm, midst.bigW, midst.bigL, midst.bigU, prms.O, midst.p, midst.y )

			self.st.y = self.st.y + midst.ydot*dt/3.0
			self.st.p = self.st.p + midst.pdot*dt/3.0
			self.st.t += dt/3.0

			#== FOURTH STEP OF RK4 ==========
			midst.y = self.ost.y + dt*midst.ydot
			midst.p = self.ost.p + dt*midst.pdot
			midst.ovm, midst.bigL, midst.bigW, midst.bigU \
					= system_fortran.update_sums( prms.Wk, prms.gk, prms.O, midst.y )

			midst.pdot, midst.ydot = system_fortran.calc_derivatives_2( prms.w_qb, prms.wd, prms.A_d, midst.t, prms.Wk, prms.gk, midst.ovm, midst.bigW, midst.bigL, midst.bigU, prms.O, midst.p, midst.y )

			self.st.y = self.st.y + midst.ydot*dt/6.0
			self.st.p = self.st.p + midst.pdot*dt/6.0
			self.st.t += dt/6.0
			self.st.ovm, self.st.bigL, self.st.bigW, self.st.bigU \
					= system_fortran.update_sums( prms.Wk, prms.gk, prms.O, self.st.y )

		elif prms.device == 'QUTR':

			pdot, fdot= system_fortran.calc_derivatives_qtr( prms.w_qb, prms.wd, prms.A_d, self.st.t, prms.Wk, prms.g_qc, prms.gk, self.st.ovm, self.st.bigW, self.st.bigL, self.st.p, self.st.y )

			self.st.y = self.st.y + fdot*dt/6.0
			self.st.p = self.st.p + pdot*dt/6.0
			self.st.t += dt/6.0

			#== SECOND self.st.P OF RK4 ==========
			midst.y = self.ost.y + 0.5*dt*fdot
			midst.p = self.ost.p + 0.5*dt*pdot
			midst.t = self.ost.t + 0.5*dt
			midst.ovm, midst.bigL, midst.bigW \
					= system_fortran.update_sums_qtr( prms.Wk, prms.gk, midst.y )

			pdot, fdot = system_fortran.calc_derivatives_qtr( prms.w_qb, prms.wd, prms.A_d, midst.t, prms.Wk, prms.g_qc, prms.gk, midst.ovm, midst.bigW, midst.bigL, midst.p, midst.y )

			self.st.y = self.st.y + fdot*dt/3.0
			self.st.p = self.st.p + pdot*dt/3.0
			self.st.t += dt/3.0

			#== THIRD self.st.P OF RK4 ==========
			midst.y = self.ost.y + 0.5*dt*fdot
			midst.p = self.ost.p + 0.5*dt*pdot
			midst.t = self.ost.t + 0.5*dt
			midst.ovm, midst.bigL, midst.bigW \
					= system_fortran.update_sums_qtr( prms.Wk, prms.gk, midst.y )

			pdot, fdot = system_fortran.calc_derivatives_qtr( prms.w_qb, prms.wd, prms.A_d, midst.t, prms.Wk, prms.g_qc, prms.gk, midst.ovm, midst.bigW, midst.bigL, midst.p, midst.y )

			self.st.y = self.st.y + fdot*dt/3.0
			self.st.p = self.st.p + pdot*dt/3.0
			self.st.t += dt/3.0

			#== FOURTH self.st.P OF RK4 ==========
			midst.y = self.ost.y + dt*fdot
			midst.p = self.ost.p + dt*pdot
			midst.t = self.ost.t + dt
			midst.ovm, midst.bigL, midst.bigW \
					= system_fortran.update_sums_qtr( prms.Wk, prms.gk, midst.y )

			pdot, fdot = system_fortran.calc_derivatives_qtr( prms.w_qb, prms.wd, prms.A_d, midst.t, prms.Wk, prms.g_qc, prms.gk, midst.ovm, midst.bigW, midst.bigL, midst.p, midst.y )

			self.st.y = self.st.y + fdot*dt/6.0
			self.st.p = self.st.p + pdot*dt/6.0
			self.st.t += dt/6.0
			self.st.ovm, self.st.bigL, self.st.bigW \
					= system_fortran.update_sums_qtr( prms.Wk, prms.gk, self.st.y )

	def error( self, prms):

		tmp1 = 0
		tmp2 = 0
		tmp3 = 0
		tmp4 = 0

		st = self.st
		ost = self.ost
		oost = self.oost

		if prms.device == 'QUTR':
			return 0

		def At():
			return prms.A_d*np.cos( prms.wd*ost.t )

		p = ost.p[:,:]
		pc = np.conj(p)
		y = ost.y[:,:,:]
		yc = np.conj(y)
		pdot = ost.pdot[:,:]
		ydot = ost.ydot[:,:,:]
		pdotc = np.conj(pdot)
		ydotc = np.conj(ydot)
		ovm = ost.ovm
		val = 0

		nl = len( p[:,0] )
		ncs = len( p[0,:] )
		nmodes = len( ost.y[0,0,:]  )

		kap = np.zeros( (nl,ncs,nl,ncs), dtype='complex128' )
		for s in range( nl ):
			for m in range( ncs ):
				for l in range( nl ):
					for n in range( ncs ):
						kap[ s,m,l,n ] = np.sum( ydot[s,m,:]*yc[s,m,:]\
								+ ydotc[s,m,:]*y[s,m,:] \
								- 2*yc[l,n,:]*ydot[s,m,:] ) 
		for i in range( prms.nl ):

			ovmr = ost.ovm[i,:,i,:]

			opdd = ( st.pdot[i,:] - oost.pdot[i,:] )/(st.t-oost.t)
			oydd = ( st.ydot[i,:,:] - oost.ydot[i,:,:] )/(st.t-oost.t)

			for m in range( ncs ):
				for n in range( ncs ):
					#==== tmp1 cajcujation
					tmp1 += ovmr[m,n]*( \
							+ pdotc[i,m]*pdot[i,n] \
							- 0.5 * pdotc[i,m]*p[i,n]*kap[i,n,i,m] \
							- 0.5 * pc[i,m]*pdot[i,n]*np.conj(kap[i,m,i,n]) \
							+ pc[i,m]*p[i,n]*( np.sum( ydotc[i,m,:]*ydot[i,n,:] )\
							+ 0.25*np.conj(kap[i,m,i,n])*kap[i,n,i,m]\
							)
							)

					#==== tmp4 cajcujation
					tmp4 += pc[i,m]*ovmr[m,n]*( \
							+ opdd[n] \
							- pdot[i,n]*kap[i,n,i,m] \
							+ p[i,n]*( np.sum( yc[i,m,:]*oydd[n,:]\
							- 0.5*( y[i,n,:]*np.conj(oydd[n,:])\
							+ yc[i,n,:]*oydd[n,:]\
							+ 2*ydotc[i,n,:]*ydot[i,n,:] ) )\
							+ 0.25*kap[i,n,i,m]**2 )
							)

					#==== tmp2 cajcujation
					tmp2 +=  pdotc[i,m]*p[i,n]*ovm[i,m,i,n]*( \
							+ prms.w_qb[i]\
							+ ost.bigW[i,m,n]  \
							+ At()*ost.bigU[i,m,n] ) \
							+ pc[i,m]*p[i,n]*ovmr[m,n]*( \
							- 0.5*np.conj(kap[i,m,i,n])\
							*( prms.w_qb[i]+ost.bigW[i,m,n]+At()*ost.bigU[i,m,n] )\
							+ np.sum( prms.Wk[:]*ydotc[i,m,:]*y[i,n,:] \
							+ At()*prms.O[0,:]*ydotc[i,m,:] ) ) 

					#tmp2_B += - prms.A_d*np.sin(prms.wd*ost.t)*prms.wd\
							#                * pc[i,m]*p[i,n]*ovmr[m,n]*ost.bigU[i,m,n]

					for j in range( nl ):
						tmp2 += pdotc[i,m]*p[j,n]*ovm[i,m,j,n]*ost.bigL[i,m,j,n] \
								+ pc[i,m]*p[j,n]*ovm[i,m,j,n]*( \
								- 0.5*np.conj(kap[i,m,j,n])*ost.bigL[i,m,j,n]  \
								+ np.sum( prms.gk[i,j,:]*ydotc[i,m,:] ) )

					#==== tmp3 cajcujation
					tmp3 += pc[i,m]*p[i,n]*ovm[i,m,i,n]*( (prms.w_qb[i] \
							+ ost.bigW[i,m,n] \
							+ At()*ost.bigU[i,m,n] )**2 \
							+ np.sum( prms.Wk[:]**2*yc[i,m,:]*y[i,n,:]  \
							+ prms.Wk[:]*At()*prms.O[0,:]*(yc[i,m,:]+y[i,n,:]) \
							+ At()**2*(prms.O[0,:])**2 ) )

					for j in range( nl ):
						tmp_sum = 0
						for pp in range( nmodes ):
							tmp_sum += ( yc[i,m,pp] + y[j,n,pp] )\
									* np.sum( ( prms.g2mat[j,i,pp,:] \
									+ 2*At()*prms.O[0,pp]*prms.gk[i,j,:] )\
									* ( yc[i,m,:] + y[j,n,:] ) )

						tmp3 += pc[i,m]*p[j,n]*ovm[i,m,j,n]*( tmp_sum \
								+ np.sum( np.diagonal( prms.g2mat[i,j,:,:] ) ) \
								+ 2*At()*np.sum( prms.O[0,:]*prms.gk[i,j,:] ) \
								+ ( prms.w_qb[i]+prms.w_qb[j] )*ost.bigL[i,m,j,n] \
								+ 2*np.sum( prms.Wk*yc[i,m,:]*y[j,n,:] )*ost.bigL[i,m,j,n] \
								+ np.sum( prms.gk[i,j,:]\
								*prms.Wk[:]*( y[j,n,:]+yc[i,m,:] ) ) )

		out =  ( -0.5*np.real(tmp4) + 0.5*tmp1 - 2*np.imag(tmp2) + tmp3 ) / ost.energy(prms)**2

		return out

	def save_dynamics(self, prms):

		for i in range(len(self.st_hist)):
			self.st_hist[i].save_state( prms, str(i) )

		folder_name = prms.save_dir+'/'+prms.param_char()
		np.savetxt( folder_name+'/TIMES_'+prms.param_char()+'.d', self.times )
		np.savetxt( folder_name+'/ERROR_'+prms.param_char()+'.d', self.error_hist )

	def load_dynamics(self, prms):

		folder_name = prms.save_dir+'/'+prms.param_char()
		times_nparray = np.loadtxt( folder_name+'/TIMES_'+prms.param_char()+'.d' )
		error_hist_nparray = np.loadtxt( folder_name+'/ERROR_'+prms.param_char()+'.d' )

		if prms.device == 'TRSM':
			for i in range( len( times_nparray ) ):
				st = state( prms, times_nparray[i], str(i) )
				self.st_hist.append( st )
		elif prms.device == 'QUTR':
			for i in range( len( times_nparray ) ):
				st = state_quantromon( prms, times_nparray[i], str(i) )
				self.st_hist.append( st )

		for i in range( len( times_nparray ) ):
			self.times.append( times_nparray[i] )
			if prms.calc_error:	
				self.error_hist.append( error_hist_nparray[i] )

		self.st = deepcopy( self.st_hist[-1] )
		self.ost = deepcopy( self.st_hist[-2] )
		self.oost = deepcopy( self.st_hist[-3] )

	def plot_population(self, prms, lvls=[0] ):

		lvl_pop_traj = []

		for i in range( len(self.st_hist) ):
			lvl_pops = [ self.st_hist[i].lvl_occupation(l) for l in range( prms.nl ) ]
			lvl_pop_traj.append( lvl_pops )

		lvl_pop_traj = np.array(lvl_pop_traj)

		fig,ax = plt.subplots()
		for l in lvls:
			ax.plot( self.times[:], lvl_pop_traj[:,l], label='|'+str(l)+'>' )

		#plt.title('POPULATIONS --'+ prms.param_char(line_break=True))

		if lvls == [0]:
			minval = lvl_pop_traj[:, lvls[0] ].min()
			maxval = lvl_pop_traj[:, lvls[0] ].max()
			for i in range( 1,8 ):
				interval = 10**(-i)
				interval_nb = int( (maxval-minval)/interval )
				if interval_nb < 2:
					continue
				else:
					break

			ticks = np.arange( 1 - (interval_nb+1)*interval, maxval+interval, interval )
			ax.set_yticks( ticks )
			ticklabels = []
			for j in range( len(ticks) ):
				ticklabels.append( ("{:."+str(i)+"f}").format( ticks[j] ) )
			ax.set_yticklabels( ticklabels )

		ax.grid()
		ax.set_xlabel(r't')
		plt.legend()
		plt.savefig( '../auto_figures/POPULATIONS_'+prms.param_char()+'.pdf', format='pdf' )
		plt.show()

		out_arr = np.zeros( ( len(self.times), prms.nl+1 ), dtype='float64' )
		out_arr[:,0] = np.array( self.times )
		for l in range(prms.nl): 
			out_arr[:,1+l] = np.real( lvl_pop_traj[:,l] )

		np.savetxt( prms.data_dir_path+'/PPLT_'+prms.param_char()+'.d', out_arr )

	def plot_population_fft(self, prms, lvls=[0] ):

		lvl_pop_traj = []
		for i in range( len(self.st_hist) ):
			lvl_pops = [ self.st_hist[i].lvl_occupation(l) for l in range( prms.nl ) ]
			lvl_pop_traj.append( lvl_pops )

		lvl_pop_traj = np.array(lvl_pop_traj)

		N = len(self.times) # Number of sample points
		T = (self.times[-1]-self.times[0])/N # sample spacing
		freqs = fft.fftfreq(N, T)[:N//2]
		for l in lvls:
			fft0 = fft.fft( lvl_pop_traj[:,l] )[:N//2]
			fft0[0]=0
			plt.plot( freqs, np.abs(fft0)**2, label='|'+str(l)+'>' )
		plt.xlim(0,20)

		plt.title('POP FFT --'+ prms.param_char(line_break=True))
		plt.xlabel(r'frequencies (GHz)')
		plt.legend()
		plt.show()

	def plot_XP_full(self, prms, s=0, xmin=None, xmax=None, fit=False, wc_guess=None):

		X_traj = []
		P_traj = []
		for i in range( len(self.st_hist) ):
			X_traj.append( self.st_hist[i].X_full( prms ,s ) ) 
			P_traj.append( self.st_hist[i].P_full( prms ,s ) ) 

		X_traj = np.array(X_traj )
		P_traj = np.array(P_traj )

		plt.title('X,P vs t --'+ prms.param_char(line_break=True))

		plt.scatter( self.times, X_traj, s=5 )
		plt.plot( self.times, X_traj )
		if xmin != None:
			plt.xlim( xmin, xmax )
		plt.xlabel(r'Time ')

		if fit:
			from scipy.optimize import curve_fit

			def function( t, amp, omega ):
				return amp*np.cos( t*omega ) 

			popt = curve_fit( function, self.times, X_traj, p0=[prms.cav_ini*2, wc_guess] )
			times_dense = np.linspace( self.times[0], self.times[-1], 100000 )
			plt.plot( times_dense, function( times_dense, popt[0][0], popt[0][1] ) )
			print('=== X quadrature fitting ===' )
			print('-- fitted frequency = {:.5f} GHz'.format( (popt[0][1]/(2*np.pi) ) ) )

		plt.show()

		
		#plt.title('P vs X --'+ prms.param_char(line_break=True))
		#indm = 50
		#plt.plot(  X_traj[0:indm], P_traj[0:indm] )
		#plt.show()

	def plot_XP(self, prms, lvl=None, s=0, tmin=None, tmax=None, fit=False, wc_guess=None, plotvstime=True):

		X_traj = []
		P_traj = []
		nb_pts = len(self.st_hist)
		for i in range( nb_pts ):
			if lvl != None:
				X_traj.append( self.st_hist[i].X( prms, lvl ,s ) ) 
				P_traj.append( self.st_hist[i].P( prms, lvl ,s ) ) 
			else:
				X_traj.append( self.st_hist[i].X_full( prms ,s ) ) 
				P_traj.append( self.st_hist[i].P_full( prms ,s ) ) 

		X_traj = np.array(X_traj )
		P_traj = np.array(P_traj )

		if plotvstime:
			plt.title('X,P vs t --'+ prms.param_char(line_break=True))
			plt.scatter( self.times, X_traj, s=5 )
			plt.plot( self.times, X_traj )
			if tmin != None:
				plt.xlim( tmin, tmax )
			plt.xlabel(r'Time ')

			if fit:
				from scipy.optimize import curve_fit

				def function( t, amp, omega ):
					return amp*np.cos( t*omega ) 

				popt = curve_fit( function, self.times, X_traj, p0=[prms.cav_ini*2, wc_guess] )
				times_dense = np.linspace( self.times[0], self.times[-1], 100000 )
				#plt.plot( times_dense, function( times_dense, popt[0][0], popt[0][1] ) )
				plt.plot( times_dense, function( times_dense, 2, 6.7826*2*np.pi ) )
				print('=== X quadrature fitting ===' )
				print('-- fitted frequency = {:.5f} GHz'.format( (popt[0][1]/(2*np.pi) ) ) )

			plt.show()

		else:
			fig, ax = plt.subplots(1)
			ax.set_title('X,P --'+ prms.param_char(line_break=True))
			if tmin != None:
				tind_min = int( (nb_pts-1)*tmin/self.times[-1])
				tind_max = int( (nb_pts-1)*tmax/self.times[-1])
			else:
				tind_min = 0
				tind_max = nb_pts-1
			ax.set_aspect('equal')
			ax.plot( X_traj[tind_min:tind_max], P_traj[tind_min:tind_max] )
			plt.show()


		
		#plt.title('P vs X --'+ prms.param_char(line_break=True))
		#indm = 50
		#plt.plot(  X_traj[0:indm], P_traj[0:indm] )
		#plt.show()

	def plot_X_fft(self, prms, l, fit = False, xmin=None, xmax=None ):

		X_traj = []
		for i in range( len(self.st_hist) ):
			X_traj.append( self.st_hist[i].X( prms, l ,0 ) ) 
		X_traj = np.array(X_traj )

		N = len(self.times) # Number of sample points
		T = (self.times[-1]-self.times[0])/N # sample spacing
		#T = 200/N # sample spacing
		freqs = fft.fftfreq(N, T)[:N//2]
		fft0 = fft.fft( X_traj )[:N//2]
		fft0_abs2 = np.real( np.abs( fft0 )**2 )
		ind = np.argmax( fft0_abs2 )
		print('=== X FFT ===' )
		print('-- freq spacing = {:.2f} MHz'.format( (freqs[1]-freqs[0])*1000 ) )
		print('-- max amp freq = {:.5f} GHz'.format( freqs[ind] ) )


		if fit:
			from scipy.optimize import curve_fit

			def gaussian(f,A,f0,gam):
				return A*np.exp( -(f-f0)**2/gam**2  ) 

			A_g = np.max( fft0_abs2 )
			popt = curve_fit( gaussian, freqs, fft0_abs2, p0=[A_g, freqs[ind], freqs[1]-freqs[0]] )

		freqs_fit = np.linspace( freqs[0], freqs[-1], len(freqs)*10 )
		plt.scatter( freqs, fft0_abs2 )
		plt.title('X FFT')
		plt.xlabel(r'frequencies (GHz)')
		#plt.xlim( prms.w_c/(2*np.pi)-0.1,prms.w_c/(2*np.pi)+0.1 )
		if xmin != None:
			plt.xlim( xmin, xmax )
		if fit:
			plt.plot( freqs_fit, gaussian(freqs_fit, popt[0][0],popt[0][1],popt[0][2]),color='red', label='fit' )
		plt.show()

		#plt.scatter( freqs, fft0_abs2 )
		#plt.title('X FFT')
		#plt.xlabel(r'frequencies (GHz)')
		##plt.xlim( prms.w_c/(2*np.pi)-0.1,prms.w_c/(2*np.pi)+0.1 )
		#plt.xlim( prms.w_c/(2*np.pi)-2,prms.w_c/(2*np.pi)+2 )
		#if fit:
		#	plt.plot( freqs, gaussian(freqs, popt[0][0],popt[0][1],popt[0][2]),color='red', label='fit' )
		#plt.ylim( 0,1000 )

	def plot_y0_fft(self, prms, l, n ):

		y0_traj = []
		for i in range( len(self.st_hist) ):
			y0_traj.append( np.real( self.st_hist[i].y_OB_lns( prms, l,n,0 ) ) ) 

		y0_traj = np.array(y0_traj )

		#plt.title('Y0 vs t --'+ prms.param_char(line_break=True))
		#
		#plt.plot( self.times, y0_traj )
		##plt.xlim( 0,1 )
		#plt.xlabel(r'Time ')
		#plt.show()


		N = len(self.times) # Number of sample points
		T = (self.times[-1]-self.times[0])/N # sample spacing
		freqs = fft.fftfreq(N, T)[:N//2]
		fft0 = fft.fft( y0_traj )[:N//2]
		#fft0[0]=0
		plt.plot( freqs, np.abs(fft0)**2 )
		plt.xlim( prms.w_c/(2*np.pi)-0.5, prms.w_c/(2*np.pi)+0.5 )

		#plt.title('Y0 FFT --'+ prms.param_char(line_break=True))
		plt.title('Y0 FFT')
		plt.xlabel(r'frequencies (GHz)')
		plt.show()

	def plot_norm(self, prms, xmin=None, xmax=None ):

		norm_traj = []
		for i in range( len(self.st_hist) ):
			norm_traj.append(self.st_hist[i].norm())

		fig,ax = plt.subplots()
		ax.plot( self.times, np.array(norm_traj), color='red', label ='norm' )
		if xmin:
			plt.xlim(xmin,xmax)
		plt.xlabel(r't')
		plt.legend()
		plt.show()
		
	def plot_energy(self, prms, xmin=None, xmax=None ):

		en_traj = []
		for i in range( len(self.st_hist) ):
			en_traj.append( self.st_hist[i].energy(prms) )

		fig,ax = plt.subplots()
		ax.plot( self.times, np.array(en_traj), color='red', label ='energy' )
		if xmin:
			plt.xlim(xmin,xmax)
		plt.xlabel(r't')
		plt.legend()
		plt.show()

	def plot_photon_number(self, prms, mint=None, maxt=None, miny=None, maxy=None ):

		cav_ph_traj = []
		for i in range( len(self.st_hist) ):
			cav_ph_traj.append( self.st_hist[i].sig_mode_ph_nb( prms, 0 ) ) 

		plt.plot( self.times, cav_ph_traj , label=r'$\langle \hat n \rangle$'  )
		if mint != None:
			plt.xlim( mint, maxt )
		if miny != None:
			plt.ylim( miny, maxy )
		plt.xlabel(r't')
		plt.title('CAVITY PHOTONS -- '+prms.param_char(line_break=True))
		plt.savefig( '../auto_figures/CAVITY_'+prms.param_char()+'.pdf', format='pdf' )
		plt.legend()
		plt.show()

		out_arr = np.zeros( ( len(self.error_hist), 2 ), dtype='float64' )
		out_arr[:,0] = np.array( self.times )
		out_arr[:,1] = np.real(np.array(cav_ph_traj))

		np.savetxt( prms.data_dir_path+'/PHOTONS_'+prms.param_char()+'.d', out_arr )

	def plot_error(self, prms, mint=None, maxt=None ):

		fig,ax = plt.subplots()
		ax.plot( self.times[:], np.imag(np.array(self.error_hist)), label ='imag' )
		ax.plot( self.times[:], np.real(np.array(self.error_hist)), label ='real' )
		plt.title('error')
		plt.xlabel(r'$t$')
		if mint != None:
			min_ind = int( mint/(self.times[1] - self.times[0]) )
			plt.xlim(mint,maxt)
			plt.ylim(0, max(np.real(np.array(self.error_hist[ min_ind: ])) )  )
		plt.legend()
		plt.show()

		out_arr = np.zeros( ( len(self.error_hist), 2 ), dtype='float64' )
		out_arr[:,0] = np.array( self.times )
		out_arr[:,1] = np.real(np.array(self.error_hist))

		np.savetxt( prms.data_dir_path+'/ERROR_'+prms.param_char()+'.d', out_arr )

	def plot_spectrum(self, prms, fit=True, minf=None, maxf=None ):

		if prms.device == 'TRSM':
			wk_arr = prms.wk[1:]
		elif prms.device == 'QUTR':
			wk_arr = prms.Wk[1:]

		spectrum = np.zeros( (prms.nmodes), dtype='float64' )
		for p in range( 0,prms.nmodes ):
			spectrum[p] = self.st_hist[-1].sig_mode_ph_nb( prms, p )
		ind = np.argmax( spectrum[1:] )

		print('-- max_freq = ', wk_arr[ind]/(2*np.pi))

		if fit:
			from scipy.optimize import curve_fit

			def lorentzian(w,A,w0,gam):
				return A / ( 1 + ((w-w0)/gam)**2 )

			A_g = np.max(spectrum)
			popt = curve_fit( lorentzian, wk_arr, spectrum[1:], p0=[A_g,prms.w_c, 0.003*2*np.pi] )


			plt.scatter( wk_arr/(2*np.pi), spectrum[1:],s=5, color='black',label='data' )
			plt.plot( wk_arr/(2*np.pi), lorentzian(wk_arr,popt[0][0],popt[0][1],popt[0][2]),color='red', label='fit' )

			print('-- center frequency/2pi: = ', popt[0][1]/(2*np.pi))
			print('-- HWHM/2pi: = ', popt[0][2]/(2*np.pi))
		else:
			plt.plot( wk_arr/(2*np.pi), spectrum[1:], linewidth=0.5,color='blue',label='data' )
			plt.scatter( wk_arr/(2*np.pi), spectrum[1:],s=5, color='black',label='data' )

		if minf:
			plt.xlim( minf, maxf )

		plt.title('SPECTRUM -- '+prms.param_char(line_break=True))
		plt.savefig( '../auto_figures/SPECTRUM_'+prms.param_char()+'.pdf', format='pdf' )
		plt.legend()
		plt.show()

	def plot_wigner(self, prms, xmin ):

		xmax=-xmin
		wigner = system_fortran.calc_wigner( self.st_hist[-1].p, \
				self.st_hist[-1].y_OB( prms, 0, rotating_frame=True ), \
				self.st_hist[-1].ovm, xmin, xmax, 100 )
		plt.axhline(y=0, dashes=[2,2,2,2])
		plt.axvline(x=0, dashes=[2,2,2,2])
		plt.imshow( wigner, extent=[xmin,xmax,xmin,xmax],origin ='lower' )
		plt.show()

	def plot_displacements(self, prms, xmin, tmin=None, tmax=None, lvls=[0,1] ):

		ncs = self.st_hist[-1].ncs
		y0_traj_2 = [ [ [] for n in range(ncs) ] for s in range(prms.nl)  ]
		nb_pts = len(self.st_hist)
		for i in range( nb_pts ):
			y_OB = self.st_hist[i].y_OB( prms, 0, rotating_frame=False ) 
			for s in range( prms.nl ):
				for n in range( self.st.ncs ):
					y0_traj_2[s][n].append( y_OB[s,n] )

		xmax=-xmin
		if tmin != None:
			tind_min = int( (nb_pts-1)*tmin/self.times[-1])
			tind_max = int( (nb_pts-1)*tmax/self.times[-1])
		else:
			tind_min = 0
			tind_max = nb_pts-1
		nl_plot = len( lvls )
		fig, axes = plt.subplots(1,nl_plot,figsize=((12/3)*nl_plot, (10/3)*nl_plot))
		for s in range(nl_plot):
			#axes[s].set_xlim( xmin, xmax )
			#axes[s].set_ylim( xmin, xmax )
		    axes[s].axhline(y=0, dashes=[2,2,2,2], color='black')
		    axes[s].axvline(x=0, dashes=[2,2,2,2], color='black')
		    for n in range( self.st.ncs ):
		        axes[s].set_title( '|'+str(s)+'>' )
		        axes[s].plot( np.real( y0_traj_2[s][n][tind_min:tind_max] ),\
		        		np.imag( y0_traj_2[s][n][tind_min:tind_max] ), color='C'+str(n),\
		                           label='n='+str(n)  )
		    axes[s].set_aspect('equal')
		    axes[s].legend()
		plt.show()

	def plot_weights_p2(self, prms, mint=None, maxt=None, lvl=0 ):

		p2_traj = []
		for i in range( len(self.st_hist) ):
			p2_traj.append( np.abs(self.st_hist[i].p) )

		ncs = self.st_hist[-1].ncs
		p2_traj_2 = [ [ [] for n in range(ncs) ] for s in range(prms.nl)  ]

		for i in range( len(self.st_hist) ):
			for s in range( prms.nl ):
				for n in range( self.st.ncs ):
					if n >= len( self.st_hist[i].p[s,:] ):
						p2_traj_2[s][n].append( -0.1 )
					else:
						p2_traj_2[s][n].append( np.abs( self.st_hist[i].p[s,n] )   )

		if mint != None:
			plt.xlim( mint, maxt )

		for n in range( ncs ):
		    plt.plot( self.times[  : ], p2_traj_2[lvl][n][:],  label='l='+str(lvl)+', n='+str(n))
		plt.legend()
		plt.savefig( '../auto_figures/WEIGHTS2_'+prms.param_char()+'.pdf', format='pdf' )
		plt.show()

		out_arr = np.zeros( ( len(self.times), ncs*(prms.nl)+1 ), dtype='float64' )
		out_arr[:,0] = np.array( self.times )
		for l in range(prms.nl): 
			for n in range(ncs): 
				out_arr[:,1+ncs*l+n] = np.real( p2_traj_2[l][n][:] )

		np.savetxt( prms.data_dir_path+'/WEIGHTS2_'+prms.param_char()+'.d', out_arr )

	def fit_decay_and_plot(self, prms, target_decay ):

		from scipy.optimize import curve_fit
		times_npa = np.array( self.times )
		def cavity_photons( t, gamma ):
		    return 1*( np.exp( -gamma*t ) )

		cav_ph_traj = []
		for i in range( len(self.st_hist) ):
			cav_ph_traj.append( self.st_hist[i].sig_mode_ph_nb( prms, 0 ) ) 
		
		popt = curve_fit( cavity_photons, times_npa , cav_ph_traj, p0=[0.001] )

		print('-- decay rate/2*np.pi: = ', popt[0]/(2*np.pi) )
		print('-- target - current: = ', target_decay - popt[0] )

		print([target_decay])
		print(popt[0])
		plt.scatter( times_npa, cav_ph_traj, s=1, color='black', label='data' )
		plt.plot( self.times, cavity_photons( times_npa, popt[0] ), label='fit' )
		plt.plot( self.times, cavity_photons( times_npa, target_decay ), label='target' )
		plt.legend()





	#def check_time_step(prms,st,ost,oost,dt,errorlimit):
	#
	#    st_tofix = deepcopy( st )
	#    oost_tofix = deepcopy( oost )
	#    deltaE = np.abs( st_tofix.energy(prms) - oost_tofix.energy(prms) )
	#    slowfactorFix = 1.
	#    fixtry = 1
	#    slowfactor = 0.5
	#
	#    while ( (np.abs(deltaE) > errorlimit) and (fixtry < 20) ):
	#        
	#        fixtry += 1
	#        slowfactor *= 2        #-- keep track of the number of tries
	#        st = deepcopy( oost_tofix )		#-- rewind evolution by 2 steps
	#        print(st.t, fixtry, slowfactor, deltaE)
	#        
	#        #if (slowfactor < 100) then
	#		  #else 
	#		# slowfactorFix = dble(slowfactor)/dble(fixtry+1)  !-- first try to decrease dt
	#		  #end if
	#            
	#        #-- RE_EVOLVE
	#        while (st_tofix.t > st.t):
	#    
	#            evolve_RK4( prms, st, ost, oost, dt/slowfactor )
	#		    #write(*,'(I10)',advance='no') int( slowfactorFix )
	#
	#	    #-- recalculate the energy error
	#        deltaE = np.abs( st.energy(prms) - oost_tofix.energy(prms) )
	#
	#    if ( (fixtry != 1) and (np.abs(deltaE) < errorlimit) ):
	#
	#        #print( "No fix for ",(cnt)," steps" )
	#        print("SUCCESSFUL, fix attempts:"\
			#	   	 ,(fixtry-1)," time=",np.round(st.t,5) ," New_deltaE=",np.round(deltaE,8)," Ini_deltaE=", np.round(ini_deltaE,8) )
	#        print()
	#    
	#    elif ( (fixtry >= 20) and (np.abs(deltaE) > errorlimit) ):
	#        print("ABORT -- error in the energy is too high")
	#        
	#        #if (np.abs(deltaE) > 1.0e-5):
	#            
	#    elif ( (fixtry == 1) and (np.abs(deltaE) < errorlimit) ):
	#        #-- count the number of steps betweeen 2 fixings
	#        pass
	#        #cnt += 1



	#def evolve_RK4_python( prms, st, ost, oost, dt ):
	#
	#    oost = deepcopy( ost )
	#    ost = deepcopy( st ) 
	#    midst = deepcopy( st )
	#
	#    #== FIRST STEP OF RK4 ==========
	#    st.calc_derivatives(prms)
	#    #system_fortran.calc_derivatives(  prms.w_qb, prms.wk, prms.gk,\
			#    #                                st.ovm, st.bigW, st.bigL,\
			#    #print('HCMS:',system_fortran.has_column_major_storage(pdot_out) )
	#    #outv=system_fortran.calc_derivatives_2(prms.nmodes, prms.nl, prms.w_qb, prms.wk, st.ncs)
	#    #outv1 = system_fortran.calc_derivatives_2(\
			#        #prms.nmodes,st.ncs, prms.nl,prms.nmodes,st.ncs, prms.nl, prms.w_qb, prms.wk)
	#    #st.calc_derivatives(prms)
	#    #pdot, fdot, inv_ov_ff, unitmat, dedpc, dedyc, bigP, bigF, d, RHS, mat2D, alphaT, val1= system_fortran.calc_derivatives_2( prms.w_qb, prms.wk, prms.gk, st.ovm, st.bigW, st.bigL, st.p, st.y )
	#    
	#    
	#
	#
	#    st.y = st.y + st.ydot*dt/6
	#    st.p = st.p + st.pdot*dt/6
	#    st.update_sums(prms)
	#
	#    #== SECOND STEP OF RK4 ==========
	#    midst.y = ost.y + 0.5*dt*st.ydot
	#    midst.p = ost.p + 0.5*dt*st.pdot
	#    midst.update_sums(prms)
	#
	#    midst.calc_derivatives(prms)
	#
	#    st.y = st.y + midst.ydot*dt/3
	#    st.p = st.p + midst.pdot*dt/3
	#    st.update_sums(prms)
	#
	#    #== THIRD STEP OF RK4 ==========
	#    midst.y = ost.y + 0.5*dt*midst.ydot 
	#    midst.p = ost.p + 0.5*dt*midst.pdot 
	#    midst.update_sums(prms)
	#
	#    midst.calc_derivatives(prms)
	#
	#    st.y = st.y + midst.ydot*dt/3
	#    st.p = st.p + midst.pdot*dt/3
	#    st.update_sums(prms)
	#
	#    #== FOURTH STEP OF RK4 ==========
	#    midst.y = ost.y + dt*midst.ydot 
	#    midst.p = ost.p + dt*midst.pdot 
	#    midst.update_sums(prms)
	#
	#    midst.calc_derivatives(prms)
	#
	#    st.y = st.y + midst.ydot*dt/6
	#    st.p = st.p + midst.pdot*dt/6
	#    st.update_sums(prms)
	#
	#    st.t = st.t + dt
	#    
	#    return ost, oost

	#def evolve_euler( prms, st, ost, oost ):
	#
	#    oost = deepcopy( ost )
	#    ost = deepcopy( st ) 
	#    midst = deepcopy( st )
	#
	#    dt = prms.dt
	#
	#    #== FIRST STEP OF RK4 ==========
	#    st.calc_derivatives(prms)
	#
	#    st.y[:,:] += st.ydot[:,:]*dt
	#    st.p[:,:] += st.pdot[:,:]*dt
	#    st.update_sums(prms)
	#    
	#    st.t = st.t + dt
	#        
	#    return ost


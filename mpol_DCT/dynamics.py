import numpy as np
import sys
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from .state import state
from .fortran_module import fortran_module
from termcolor import colored
from time import time
from datetime import timedelta


class dynamics(object):

	def __init__( self, *args ):

		prms = args[0]
		if len(args) == 1:
			if prms.device[0:4] == 'TRSM':
				self.st = state( prms, prms.ncs_ini )
			elif prms.device[0:4] == 'QUTR':
				self.st = state_quantromon( prms, prms.ncs_ini )

		elif len(args) > 1:
			self.st = args[1]

		self.ost = deepcopy( self.st )

		self.times = []
		self.st_hist = []
		self.error_med_hist = []
		self.adding_times = []
		self.truncation_str = ""

	def truncate_evolution( self, time ):

		percentage = (self.st.t-time)/self.st.t
		steps = len( self.times )        

		for i in range( int( steps*percentage) ):
			self.times.pop()
			self.st_hist.pop()
			if self.error_med_hist != None:
				self.error_med_hist.pop()

		self.st = self.st_hist[-1]
		self.ost = self.st_hist[-2]

	def time_evolve( self, prms, tmax, print_delay, error_calc_delay ):

		max_max_slowfactor=1

		print_times = list( np.linspace( self.st.t-1e-7, tmax + 2*print_delay, 11 ) )
		timeline=[]

		tref = 0.0
		last_cpu_t = time()
		last_print_t = -1e10
		error_buffer = np.zeros( prms.median_pts, dtype='float64' )

		while ( self.st.t < tmax ):

			#==== printing the each 1/10 of the simulation
			if prms.time_bar:
				if print_times[0] < self.st.t:
					print_times.pop(0)
					print( 't=', "{:.1f}".format( self.st.t )+' || '\
							#+"{:9s}".format( str( timedelta(seconds=int(time()-last_cpu_t)) ) )+' || '\
							+"{:7d}".format( int( time()-last_cpu_t ) )+' || '\
							+'ncs=',"{:2d}".format( self.st.ncs )+' || ' \
							+'max_slowfactor=',"{:.0e}".format( max_max_slowfactor ) )
					max_max_slowfactor = 1
					last_cpu_t = time()


			#==== error calculcation and printing
			if prms.calc_error:
					
				#--  evolve with smaller time-step to get more accurate second derivative
				self.ost = deepcopy( self.st )
				self.ost.calculate_derivatives( prms )			
				max_max_slowfactor = self.all_fortran_time_evolve( prms, self.st.t + 0.99*prms.dt*1e-4, prms.dt*1e-4, -1e8, max_max_slowfactor, adaptive=True )
				self.st.calculate_derivatives( prms )			

				#--  calculate and store error in array in rolling array
				error = fortran_module.get_error( prms.w_qb, prms.wd, prms.A_d, self.ost.t, self.st.t, prms.Wk, prms.gk,\
						self.ost.ovm, self.ost.bigW, self.ost.bigL, self.ost.bigU, prms.O, prms.g2mat,\
						self.ost.p, self.ost.y, self.ost.pdot, self.ost.ydot, self.st.pdot, self.st.ydot,\
						self.ost.energy( prms ) )
				error_buffer = np.roll( error_buffer, -1 )
				error_buffer[-1] = np.real( error )

				#--  no need to revert to previous state
				#self.st = self.ost

			#==== data printing
			if self.st.t > last_print_t + print_delay:
				self.times.append( self.st.t )
				self.st_hist.append( deepcopy(self.st) )
				self.error_med_hist.append( np.median( error_buffer ) )
				last_print_t = self.st.t

			#==== THE EVOLUTION ROUTINE
			max_max_slowfactor = self.all_fortran_time_evolve( prms, min( self.st.t + error_calc_delay, tmax + 0.5*prms.dt ), \
										prms.dt, tref, max_max_slowfactor, adaptive=True )

			#==== ADDING COHERENT STATES WHEN NEEDED
			if self.st.ncs < prms.ncs_max \
					and self.st.t > tref + prms.dt_add \
					and self.error_med_hist[-1] > prms.error_thr_arr[ self.st.ncs-prms.ncs_ini ]:

				self.add_coherent_state( prms )
				tref = self.st.t

	def all_fortran_time_evolve( self, prms, tf, dt, tref, max_max_slowfactor, adaptive ):

		if prms.device[0:4] == 'TRSM': 

			#-- Fortran time-evolve routine
			self.st.p, self.st.y, self.st.t, max_slowfactor, info = fortran_module.time_evolve( prms.err_lim, prms.w_qb,\
					prms.wd, prms.A_d, self.st.t, tref, tf, dt, prms.Wk, prms.gk, self.st.ovm, self.st.bigW, self.st.bigL,\
					self.st.bigU, prms.O, self.st.p, self.st.y, adaptive, prms.lim_slowfactor )
			if max_slowfactor > max_max_slowfactor:
				max_max_slowfactor = max_slowfactor
			if info==-1:
				print('=========================================')
				print('== ERROR -1: value of slowfactor higher than '+str(prms.lim_slowfactor)+'.')
				print('== Abroting ==')
				print('=========================================')
				sys.exit()

			#-- Fortran update routine: updating ovm, bigL, bigW and bigU
			self.st.ovm, self.st.bigL, self.st.bigW, self.st.bigU \
					= fortran_module.update_sums( prms.Wk, prms.gk, prms.O, self.st.y )

		elif prms.device[0:4] == 'QUTR': 
			self.st.p, self.st.y, self.st.t = fortran_module.time_evolve_qtr( prms.w_qb, prms.wd, prms.A_d, self.st.t, tf,\
					prms.dt, prms.Wk, prms.g_qc, prms.gk, self.st.ovm, self.st.bigW, self.st.bigL, self.st.p, self.st.y )

			self.st.ovm, self.st.bigL, self.st.bigW \
					= fortran_module.update_sums_qtr( prms.Wk, prms.gk, self.st.y )

		return max_max_slowfactor

	def add_coherent_state( self, prms ):

		self.adding_times.append( self.st.t  )
		self.st.update_sums( prms )
		norm_before = self.st.norm()

		new_ncs = self.st.ncs+1 
		new_st = state( prms, new_ncs )

		new_st.p[:,0:new_ncs-1] = self.st.p
		new_st.p[:,new_ncs-1] = prms.p0
		new_st.y[:,0:new_ncs-1,:] = self.st.y
		
		for s in range( prms.nl ):

			n_heaviest = np.argmax( abs(self.st.p[s,:])**2 )

			#for n in range( self.st.ncs ):
			#	if np.abs(self.st.p[s,n])**2 > p2_max:
			#		p2_max = np.abs(self.st.p[s,n])**2
			#		s_heaviest = s
			#		n_heaviest = n


			#-- if ratio imaginary set new coherent state to CS-1
			#-- else just add olf heaviest times the ratio
			if np.abs( np.imag( prms.adding_ratio ) ) < 1e-9:
				new_st.y[s,new_ncs-1,:] = prms.adding_ratio*new_st.y[s,n_heaviest,:]
			else:
				#normalised_displacements = new_st.y[s,n_heaviest,:]/np.abs(new_st.y[s,n_heaviest,:])
				new_st.y[s,new_ncs-1,:] = new_st.y[s,n_heaviest,:] - 1j*prms.adding_ratio*prms.O[0,:]
				#for k in range( prms.nmodes ):

		new_st.t = self.st.t

		new_st.update_sums( prms )
		new_st.normalize()

		self.st = new_st

		norm_after = self.st.norm()

		print('====================')
		print( "CS ADDED: from ",self.st.ncs-1," to ",self.st.ncs, "at t=", self.st.t )
		print( "Delta norm =", norm_after - norm_before )
		print('====================')

	#====================================
	#== LOAD AND SAVE
	#====================================

	def save_dynamics(self, prms):

		for i in range(len(self.st_hist)):
			self.st_hist[i].save_state( prms, str(i) )

		out_error = np.real( np.array(self.error_med_hist) )

		folder_name = prms.save_dir+'/'+prms.param_char()
		np.savetxt( folder_name+'/TIMES_'+prms.param_char()+'.d', self.times )
		np.savetxt( folder_name+'/ERROR_'+prms.param_char()+'.d', out_error )

	def load_dynamics(self, prms):

		folder_name = prms.save_dir+'/'+prms.param_char()
		times_nparray = np.loadtxt( folder_name+'/TIMES_'+prms.param_char()+'.d' )
		error_hist_nparray = np.loadtxt( folder_name+'/ERROR_'+prms.param_char()+'.d' )

		if prms.device[0:4] == 'TRSM':
			for i in range( len( times_nparray ) ):
				st = state( prms, times_nparray[i], str(i) )
				self.st_hist.append( st )
		elif prms.device[0:4] == 'QUTR':
			for i in range( len( times_nparray ) ):
				st = state_quantromon( prms, times_nparray[i], str(i) )
				self.st_hist.append( st )

		for i in range( len( times_nparray ) ):
			self.times.append( times_nparray[i] )
			if prms.calc_error:	
				self.error_med_hist.append( error_hist_nparray[i] )


		self.st = deepcopy( self.st_hist[-1] )
		self.ost = deepcopy( self.st_hist[-2] )


	#====================================
	#== PLOTTING FUNCTIONS
	#====================================

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

		y0_traj = np.array( y0_traj )

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

		out_arr = np.zeros( ( len(self.times), 2 ), dtype='float64' )
		out_arr[:,0] = np.array( self.times )
		out_arr[:,1] = np.real( np.array(cav_ph_traj) )

		np.savetxt( prms.data_dir_path+'/PHOTONS_'+prms.param_char()+'.d', out_arr )

	def plot_error_OLD(self, prms, mint=None, maxt=None, miny=None, maxy=None, pt_avg=10, plot_used_error=False ):

		fig,ax = plt.subplots()
		plt.title('error')
		plt.xlabel(r'$t$')

		##--  truncating too high values
		#if y_remove != None and y_remove != -1:
		#	for i in range( len(mod_error_hist) ):
		#		if np.abs(mod_error_hist[i]) > y_remove:
		#			if i>0:
		#				mod_error_hist[i] = mod_error_hist[i-1]
		#			else:
		#				mod_error_hist[i] = 0

		mod_error_hist = copy( self.error_hist )
		error_avg_hist = []

		if pt_avg > 0:
			for i in range( len(mod_error_hist) ):
				if i <= pt_avg:
					error_avg_hist.append( np.mean(mod_error_hist[:i]) )
				else:
					error_avg_hist.append( np.mean(mod_error_hist[i-pt_avg:i]) )
		elif pt_avg<0:
			for i in range( len(mod_error_hist) ):
				if i <= np.abs(pt_avg):
					error_avg_hist.append( np.median(mod_error_hist[:i]) )
				else:
					error_avg_hist.append( np.median(mod_error_hist[i-np.abs(pt_avg):i]) )

		new_times = np.linspace( self.times[0], self.times[-1], len( self.error_hist ) )
		new_med_times = np.linspace( self.times[0], self.times[-1], len( self.error_med_hist ) )

		if plot_used_error:
			ax.plot( new_med_times, np.real(np.array(self.error_med_hist)), label ='USED' )

		ax.plot( new_times, np.real(np.array(error_avg_hist)), label ='SPECIFIED' )

		if mint != None:
			min_ind = int( mint/(new_times[1] - new_times[0]) )
			plt.xlim( mint , maxt )
		plt.legend()
		if miny != None:
			plt.ylim( miny, maxy )
		elif mint != None:
			min_ind = int( mint/(new_times[1] - new_times[0]) )
			#plt.ylim(0, max(np.real(np.array(mod_error_hist[ min_ind: ])) )  )

		#if ylim_median_factor != None:
		#	from statistics import median
		#	y_median = median( error_avg_hist )

		#	plt.ylim( 0, ylim_median_factor*y_median )

		plt.show()

		len_max = max( len(self.error_hist), len(self.error_med_hist) )
		out_arr = np.zeros( ( len_max, 4 ), dtype='float64' )
		out_arr[:,0] = np.array( new_times )
		out_arr[:,1] = np.real(np.array(self.error_hist))[:len_max]
		out_arr[:,2] = np.real(np.array(error_avg_hist))[:len_max]
		out_arr[:,3] = np.real(np.array(self.error_med_hist))[:len_max]

		np.savetxt( prms.data_dir_path+'/ERROR_'+prms.param_char()+'.d', out_arr )

	def plot_error(self, prms, xmin=None, xmax=None, ymin=None, ymax=None ):

		fig,ax = plt.subplots()
		plt.title('error')
		plt.xlabel(r'$t$')

		ax.plot( self.times, np.real(np.array(self.error_med_hist)) )

		if xmin != None:
			plt.xlim( xmin , xmax )
		if xmax != None:
			plt.ylim( ymin, ymax )

		plt.show()

		out_arr = np.zeros( ( len(self.error_med_hist), 2 ), dtype='float64' )
		out_arr[:,0] = np.array( self.times )
		out_arr[:,1] = np.real( np.array(self.error_med_hist) )

		np.savetxt( prms.data_dir_path+'/ERROR_'+prms.param_char()+'.d', out_arr )

	def plot_spectrum(self, prms, fit=True, minf=None, maxf=None, ymax=None ):

		if prms.device[0:4] == 'TRSM':
			wk_arr = prms.wk[1:]
		elif prms.device[0:4] == 'QUTR':
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

		if ymax:
			plt.ylim( -0.05*ymax, ymax )


		plt.title('SPECTRUM -- '+prms.param_char(line_break=True))
		plt.savefig( '../auto_figures/SPECTRUM_'+prms.param_char()+'.pdf', format='pdf' )
		plt.legend()
		plt.show()

	def plot_wigner(self, prms, xmin, log_min=None ):

		from matplotlib.colors import LogNorm
		from matplotlib.colors import Normalize

		xmax=-xmin
		wigner = fortran_module.calc_wigner( self.st_hist[-1].p, \
				self.st_hist[-1].y_OB( prms, 0, rotating_frame=True ), \
				self.st_hist[-1].ovm, xmin, xmax, 100 )
		plt.axhline(y=0, dashes=[2,2,2,2])
		plt.axvline(x=0, dashes=[2,2,2,2])


		if log_min:
			norm_husimi=LogNorm(vmin=log_min)
			plt.imshow( wigner, extent=[xmin,xmax,xmin,xmax],norm=norm_husimi,origin ='lower' )
		else:
			plt.imshow( wigner, extent=[xmin,xmax,xmin,xmax],origin ='lower' )
		plt.colorbar()
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

		minp = 10
		maxp = 0

		for i in range( len(self.st_hist) ):
			for s in range( prms.nl ):
				for n in range( self.st.ncs ):
					if n >= len( self.st_hist[i].p[s,:] ):
						#p2_traj_2[s][n].append( -0.1 )
						p2_traj_2[s][n].append( 1e8 )
					else:
						pval = np.abs( self.st_hist[i].p[s,n] )
						p2_traj_2[s][n].append( pval )
						if minp > pval:
							minp = pval
						if pval < 0.5e8 and maxp < pval:
							maxp = pval
							
		if mint != None:
			plt.xlim( mint, maxt )

		plt.ylim( minp/2, 2*maxp )

		for n in range( ncs ):
		    plt.plot( self.times[  : ], p2_traj_2[lvl][n][:],  label='l='+str(lvl)+', n='+str(n))
		plt.legend()
		plt.savefig( '../auto_figures/WEIGHTS2_'+prms.param_char()+'.pdf', format='pdf' )
		plt.yscale('log')
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
	##-- 


#	def error( self, prms):
#
#		tmp1 = 0
#		tmp2 = 0
#		tmp3 = 0
#		tmp4 = 0
#
#		st = self.st
#		ost = self.ost
#
#		if prms.device == 'QUTR':
#			return 0
#
#		def At():
#			return prms.A_d*np.cos( prms.wd*ost.t )
#
#		p = ost.p[:,:]
#		pc = np.conj(p)
#		y = ost.y[:,:,:]
#		yc = np.conj(y)
#		pdot = ost.pdot[:,:]
#		ydot = ost.ydot[:,:,:]
#		pdotc = np.conj(pdot)
#		ydotc = np.conj(ydot)
#		ovm = ost.ovm
#		val = 0
#
#		nl = len( p[:,0] )
#		ncs = len( p[0,:] )
#		nmodes = len( ost.y[0,0,:]  )
#
#		kap = np.zeros( (nl,ncs,nl,ncs), dtype='complex128' )
#		for s in range( nl ):
#			for m in range( ncs ):
#				for l in range( nl ):
#					for n in range( ncs ):
#						kap[ s,m,l,n ] = np.sum( ydot[s,m,:]*yc[s,m,:]\
#								+ ydotc[s,m,:]*y[s,m,:] \
#								- 2*yc[l,n,:]*ydot[s,m,:] ) 
#		for i in range( prms.nl ):
#
#			ovmr = ost.ovm[i,:,i,:]
#
#			opdd = ( st.pdot[i,:] - ost.pdot[i,:] )/(st.t-ost.t)
#			oydd = ( st.ydot[i,:,:] - ost.ydot[i,:,:] )/(st.t-ost.t)
#
#			for m in range( ncs ):
#				for n in range( ncs ):
#					#==== tmp1 cajcujation
#					tmp1 += ovmr[m,n]*( \
#							+ pdotc[i,m]*pdot[i,n] \
#							- 0.5 * pdotc[i,m]*p[i,n]*kap[i,n,i,m] \
#							- 0.5 * pc[i,m]*pdot[i,n]*np.conj(kap[i,m,i,n]) \
#							+ pc[i,m]*p[i,n]*( np.sum( ydotc[i,m,:]*ydot[i,n,:] )\
#							+ 0.25*np.conj(kap[i,m,i,n])*kap[i,n,i,m]\
#							)
#							)
#
#					#==== tmp4 cajcujation
#					tmp4 += pc[i,m]*ovmr[m,n]*( \
#							+ opdd[n] \
#							- pdot[i,n]*kap[i,n,i,m] \
#							+ p[i,n]*( np.sum( yc[i,m,:]*oydd[n,:]\
#							- 0.5*( y[i,n,:]*np.conj(oydd[n,:])\
#							+ yc[i,n,:]*oydd[n,:]\
#							+ 2*ydotc[i,n,:]*ydot[i,n,:] ) )\
#							+ 0.25*kap[i,n,i,m]**2 )
#							)
#
#					#==== tmp2 cajcujation
#					tmp2 +=  pdotc[i,m]*p[i,n]*ovm[i,m,i,n]*( \
#							+ prms.w_qb[i]\
#							+ ost.bigW[i,m,n]  \
#							+ At()*ost.bigU[i,m,n] ) \
#							+ pc[i,m]*p[i,n]*ovmr[m,n]*( \
#							- 0.5*np.conj(kap[i,m,i,n])\
#							*( prms.w_qb[i]+ost.bigW[i,m,n]+At()*ost.bigU[i,m,n] )\
#							+ np.sum( prms.Wk[:]*ydotc[i,m,:]*y[i,n,:] \
#							+ At()*prms.O[0,:]*ydotc[i,m,:] ) ) 
#
#					#tmp2_B += - prms.A_d*np.sin(prms.wd*ost.t)*prms.wd\
#							#                * pc[i,m]*p[i,n]*ovmr[m,n]*ost.bigU[i,m,n]
#
#					for j in range( nl ):
#						tmp2 += pdotc[i,m]*p[j,n]*ovm[i,m,j,n]*ost.bigL[i,m,j,n] \
#								+ pc[i,m]*p[j,n]*ovm[i,m,j,n]*( \
#								- 0.5*np.conj(kap[i,m,j,n])*ost.bigL[i,m,j,n]  \
#								+ np.sum( prms.gk[i,j,:]*ydotc[i,m,:] ) )
#
#					#==== tmp3 cajcujation
#					tmp3 += pc[i,m]*p[i,n]*ovm[i,m,i,n]*( (prms.w_qb[i] \
#							+ ost.bigW[i,m,n] \
#							+ At()*ost.bigU[i,m,n] )**2 \
#							+ np.sum( prms.Wk[:]**2*yc[i,m,:]*y[i,n,:]  \
#							+ prms.Wk[:]*At()*prms.O[0,:]*(yc[i,m,:]+y[i,n,:]) \
#							+ At()**2*(prms.O[0,:])**2 ) )
#
#					for j in range( nl ):
#						tmp_sum = 0
#						for pp in range( nmodes ):
#							tmp_sum += ( yc[i,m,pp] + y[j,n,pp] )\
#									* np.sum( ( prms.g2mat[j,i,pp,:] \
#									+ 2*At()*prms.O[0,pp]*prms.gk[i,j,:] )\
#									* ( yc[i,m,:] + y[j,n,:] ) )
#
#						tmp3 += pc[i,m]*p[j,n]*ovm[i,m,j,n]*( tmp_sum \
#								+ np.sum( np.diagonal( prms.g2mat[i,j,:,:] ) ) \
#								+ 2*At()*np.sum( prms.O[0,:]*prms.gk[i,j,:] ) \
#								+ ( prms.w_qb[i]+prms.w_qb[j] )*ost.bigL[i,m,j,n] \
#								+ 2*np.sum( prms.Wk*yc[i,m,:]*y[j,n,:] )*ost.bigL[i,m,j,n] \
#								+ np.sum( prms.gk[i,j,:]\
#								*prms.Wk[:]*( y[j,n,:]+yc[i,m,:] ) ) )
#
#		out =  ( -0.5*np.real(tmp4) + 0.5*tmp1 - 2*np.imag(tmp2) + tmp3 ) / ost.energy(prms)**2
#
#		return out
#

	#def evolve_RK4_python( prms, st, ost, oost, dt ):
	#
	#    oost = deepcopy( ost )
	#    ost = deepcopy( st ) 
	#    midst = deepcopy( st )
	#
	#    #== FIRST STEP OF RK4 ==========
	#    st.calc_derivatives(prms)
	#    #fortran_module.calc_derivatives(  prms.w_qb, prms.wk, prms.gk,\
			#    #                                st.ovm, st.bigW, st.bigL,\
			#    #print('HCMS:',fortran_module.has_column_major_storage(pdot_out) )
	#    #outv=fortran_module.calc_derivatives(prms.nmodes, prms.nl, prms.w_qb, prms.wk, st.ncs)
	#    #outv1 = fortran_module.calc_derivatives(\
			#        #prms.nmodes,st.ncs, prms.nl,prms.nmodes,st.ncs, prms.nl, prms.w_qb, prms.wk)
	#    #st.calc_derivatives(prms)
	#    #pdot, fdot, inv_ov_ff, unitmat, dedpc, dedyc, bigP, bigF, d, RHS, mat2D, alphaT, val1= fortran_module.calc_derivatives( prms.w_qb, prms.wk, prms.gk, st.ovm, st.bigW, st.bigL, st.p, st.y )
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


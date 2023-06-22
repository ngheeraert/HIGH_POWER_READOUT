import numpy as np
import os
import time

twopi = 2*np.pi

class params(object):

	def __init__( self, nb_lvs=2, w_c=1.25, g_qc=0.016,gamma=0.01,\
			nbathmodes = 100, bw=0.4, bc = 1.25, wd=1.0, A_d=0.0,\
			tmax=10, dt=0.1,norm_errmax=1e-7,err_lim=1e-4,vb=False,\
			calc_error=True, error_threshold=10, p0=1e-5, ncs_ini=1,ncs_max=1,dt_add=0.1,\
			qb_ini=[0], cav_ini=0, pt_nb=300,  offset=0.1, device = 'TRSM', time_bar=False):  

		self.w_c = w_c*2*np.pi

		if device == 'TRSM':
			if 7.41*2*np.pi < self.w_c < 7.43*2*np.pi:
				energy_levels = np.loadtxt( os.path.dirname(__file__)+"/qubit_params/Elvl_Ec0.192_Ej14.155.txt" )*2*np.pi
				charge_op = np.loadtxt( os.path.dirname(__file__)+"/qubit_params/charge_op_Ec0.192_Ej14.155.txt" )
			elif 6.77*2*np.pi < self.w_c < 6.79*2*np.pi:
				energy_levels = np.loadtxt( os.path.dirname(__file__)+"/qubit_params/Elvl_Ec-0.221_wq5.4.txt" )*2*np.pi
				charge_op = np.loadtxt( os.path.dirname(__file__)+"/qubit_params/charge_op_Ec-0.221_wq5.4.txt" )
		elif device == 'QUTR':
			if 7.41*2*np.pi < self.w_c < 7.43*2*np.pi:
				energy_levels = np.loadtxt( os.path.dirname(__file__)+"/qubit_params/Elvl_QTM_Ec0.192_Ej14.155.txt" )*2*np.pi
				charge_op = np.loadtxt( os.path.dirname(__file__)+"/qubit_params/charge_op_QTM_Ec0.192_Ej14.155.txt" )
			elif 6.77*2*np.pi < self.w_c < 6.79*2*np.pi:
				energy_levels = np.loadtxt( os.path.dirname(__file__)+"/qubit_params/Elvl_QTM_Ec0.2015_Ej19.48.txt" )*2*np.pi
				charge_op = np.loadtxt( os.path.dirname(__file__)+"/qubit_params/charge_op_QTM_Ec0.2015_Ej19.48.txt" )

		self.data_dir_path = '../data'
		self.save_dir = 'trajectory_saves'
		self.nl = nb_lvs
		self.w_qb = np.zeros( self.nl, dtype='float64' )
		self.w_ge = energy_levels[1]
		self.w_qb[:self.nl] = energy_levels[:self.nl]

		self.time_bar = time_bar
		self.dt_print = 0.01
		self.calc_error = calc_error
		self.nb_of_points = pt_nb
		self.qb_ini = qb_ini
		self.cav_ini = cav_ini
		self.ratio = 0.0
		self.verbose = vb
		self.offset = offset
		self.dt_add = dt_add
		self.ncs_ini = ncs_ini
		self.ncs_max = ncs_max
		self.t_add_arr = dt_add*np.arange( 1, self.ncs_max )
		self.p0 = p0
		self.bw = bw*2*np.pi
		self.bc = bc*2*np.pi
		self.A_d = A_d*2*np.pi
		self.g_qc_val = g_qc*2*np.pi
		self.wd = wd*2*np.pi
		self.anh = (energy_levels[2]-2*energy_levels[1])
		self.gamma = gamma*2*np.pi
		self.dw = None
		self.dt = dt
		self.device = device
		self.err_lim = err_lim
		self.norm_errmax = norm_errmax
		self.norm_error_lim = 1e-7
		self.error_thr = error_threshold

		#for i in range( self.nl ):
			#self.w_qb[i] = w_qb*i - (anh/2)*i*(1-i)

		if device == 'TRSM':

			self.g_qc = self.g_qc_val*charge_op[ :self.nl, :self.nl ] \
					/ charge_op[ 0,1 ]
			self.nmodes = nbathmodes + 1
			self.wk = np.zeros( self.nmodes )
			qubit_freq_modes =\
					np.linspace( self.w_qb[1] - self.ratio*self.bw/2,\
					self.w_qb[1] + self.ratio*self.bw/2,\
					int(self.ratio*nbathmodes) )
			cavity_freq_modes =\
					np.linspace( self.bc - (1-self.ratio)*self.bw/2,\
					self.bc + (1-self.ratio)*self.bw/2,\
					nbathmodes - int(self.ratio*nbathmodes) )

			self.wk[0] = self.w_c
			self.wk[1:] = np.append( qubit_freq_modes, cavity_freq_modes )
			self.dw = self.wk[2] - self.wk[1]

			#self.g_qc = np.zeros( (self.nl,self.nl), dtype='float64' )
			#for i in range( self.nl-1 ):
			#    self.g_qc[i,i+1] = self.g_qc_val*np.sqrt(i+1) \
					#                        * ( 1 + self.anh*i/(2*self.w_qb[1]) )
			#    self.g_qc[i+1,i] = self.g_qc[i,i+1]

			h_pp = np.zeros( (self.nmodes,self.nmodes), dtype='float64' )
			for i in range( self.nmodes ):
				h_pp[i,i] = self.wk[i]
				if i>0:
					h_pp[0,i] = self.gamma * np.sqrt(self.dw)
					h_pp[i,0] = self.gamma * np.sqrt(self.dw)

			self.Wk , self.O = np.linalg.eigh( h_pp )
			self.gk = np.zeros( (self.nl,self.nl,self.nmodes) )
			for i in range( self.nl ):
				for j in range( self.nl ):
					self.gk[i,j,:] = self.O[0,:] * self.g_qc[i,j]

			self.length = np.pi/self.dw
			self.dx  = self.length/self.nmodes
			print('=== PARAMETERS ===')
			print('-- num levels   = ',self.nl)
			print('-- w01          = ',np.round(self.w_qb[1]/twopi,4) )
			print('-- wd           = ',np.round(self.wd/twopi,4))
			print('-- wc           = ',np.round(self.w_c/twopi,4))
			print('-- g_qc         = ',np.round(self.g_qc_val/twopi,4))
			print('-- p0           = ',self.p0)
			print('-- offset       = ',self.offset)
			print('-- bandwidth    = ',np.round(self.bw/twopi,4))
			print('-- band center  = ',np.round(self.bc/twopi,4))
			print('-- dt           = ',self.dt)
			print('-- A_d          = ',np.round(self.A_d/twopi,4))
			#print('')
			#print('-- hpp')
			#print(h_pp)
			#print('-- wk')
			#print(self.Wk)
			#print('-- gk')
			#print(self.gk)
			#print('-- O')
			#print(self.O)

			if self.calc_error:
				t1 = time.time()
				self.g2mat = np.zeros( (self.nl,self.nl,self.nmodes,self.nmodes),\
						dtype='float64' )
				for s in range( self.nl ):
					for i in range( self.nl ):
						for j in range( self.nl ):
							for pp in range( self.nmodes ):
								self.g2mat[i,j,pp,:] += self.gk[s,i,pp]*self.gk[s,j,:] 
				t2 = time.time()
				print('-- g2mat_generation time:')
				print(t2-t1)

		elif device == 'QUTR': 

			self.g_qc = self.g_qc_val*charge_op[ :self.nl, :self.nl ] \
					/ charge_op[ 0,2 ]
			#self.g_qc = np.zeros( (self.nl,self.nl) )
			#for i in range( 0, self.nl, 2 ):
			#	for j in range( 0, self.nl, 2 ):
			#		if i .ne. 
			#		self.g_qc[i,j] = self.g_qc_val
					

			self.nmodes = nbathmodes + 1
			self.Wk = np.zeros( self.nmodes )
			self.Wk[0] = self.w_c
			self.Wk[1:] = np.linspace( self.bc - self.bw/2,\
					self.bc + self.bw/2,\
					nbathmodes )
			self.dw = self.Wk[2] - self.Wk[1]
			self.gk = self.gamma*np.sqrt(self.dw)*np.ones( nbathmodes )

			print('=== PARAMETERS ===')
			print('-- num levels = ',self.nl)
			print('-- w01        = ',self.w_qb[1]/twopi)
			print('-- w_c        = ',self.w_c/twopi)
			#print('-- g_qc[0,1]  = ',self.g_qc[0,2]/twopi)
			print('-- gk[1]      = ',self.gk[0]/twopi)
			print('-- bandwidth  = ',self.bw/twopi)
			print('-- dt         = ',self.dt)
			print('')

	def param_char(self, line_break=False):

		out = 'nl'+str(self.nl)+'_wc'+str(np.round(self.w_c/twopi,3))\
				+'_n'+str(self.ncs_ini)\
				+'_'+str(self.ncs_max)\
				+'_wq'+str(np.round(self.w_ge/twopi,4))\
				+'_g'+str(np.round(self.g_qc_val/twopi,4))\
				+'_gam'+str(np.round(self.gamma/twopi,4))\
				+'_nm'+str(self.nmodes-1)\
				+'_p'+str(self.p0)

		if line_break:
			out += "\n"

		qbini_str = ''
		for i in range(len(self.qb_ini)):
			qbini_str = str( self.qb_ini[i] )

		if self.calc_error:
			out += '_E'+str(self.error_thr)
		else:
			out += '_NE'

		out += '_bw'+str(np.round(self.bw/twopi,4))\
				+'_bc'+str(np.round(self.bc/twopi,4))\
				+'_wd'+str(np.round(self.wd/twopi,4))\
				+'_Ad'+str(np.round(self.A_d/twopi,4))\
				+'_oft'+str(self.offset) \
				+'_anh'+str(np.round(self.anh/twopi,4))\
				+'_dt'+str(self.dt)\
				+'_errl'+str(self.err_lim)\
				+'_qb'+qbini_str\
				+'_cv'+str(self.cav_ini)\
				+'_'+str(self.device)

		return out

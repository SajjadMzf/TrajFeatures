import os
import cv2
import numpy as np 
import pickle
import h5py
import matplotlib.pyplot as plt
import read_csv as rc
import param as p
from utils import rendering_funcs as rf
import pandas
import pdb
# python -m pdb -c continue 
class RenderScenarios:
    """This class is for rendering extracted scenarios from HighD dataset 
    recording files (needs to be called seperately for each scenario).
    """
    def __init__(
        self,
        file_num:'Number of recording file being rendered',
        track_path:'Path to track file', 
        track_pickle_path:'Path to track pickle file', 
        frame_pickle_path:'Path to frame pickle file',
        static_path:'Path to static file',
        meta_path:'Path to meta file',
        dataset_name: 'Dataset  Name'):
       
        #self.metas = rc.read_meta_info(meta_path)
        self.fr_div = p.IN_FPS/p.FPS#self.metas[rc.FRAME_RATE]/p.FPS
        self.track_path = track_path
        self.scenarios = []
        self.file_num = file_num
        
        # Default Settings
        
        ''' 1. Representation Properties:'''
        self.filled = True
        self.empty = False
        self.dtype = bool
        # 1.3 Others
        
        
        self.LC_states_dir = "../Datasets/" + dataset_name + "/Scenarios"  
        self.LC_image_dataset_rdir = "../Datasets/" + dataset_name + \
            "/RenderedDataset"
        self.frames_data = rc.read_track_csv(track_path, 
                                             frame_pickle_path,
                                              group_by = 'frames', 
                                              reload = True,
                                              fr_div = self.fr_div)
        self.data_tracks = rc.read_track_csv(track_path, 
                                             track_pickle_path,
                                              group_by = 'tracks', 
                                              reload = True, 
                                              fr_div = self.fr_div)
        self.track_list = [data_track[rc.TRACK_ID][0] \
                           for data_track in self.data_tracks]
        
        #self.statics = rc.read_static_info(static_path)
        df = pandas.read_csv(track_path)
        selected_frames = (df.frame%self.fr_div == 0).tolist()
        df = df.loc[selected_frames]
        self.frame_list = [data_frame[rc.FRAME][0] \
                            for data_frame in self.frames_data]
        if p.DATASET == 'Processed_highD':
            self.metas = rc.read_meta_info(meta_path)
        self.update_dirs()
        
    def load_scenarios(self):
        file_dir = os.path.join(self.LC_states_dir,
                                 str(self.file_num).zfill(2) + '.pickle')
        with open(file_dir, 'rb') as handle:
            self.scenarios = pickle.load(handle)
    
    def save_dataset(self):
        file_dir = os.path.join(self.LC_image_dataset_dir, 
                                str(self.file_num).zfill(2) + '.h5')
        npy_dir = os.path.join(self.LC_image_dataset_dir, 
                               str(self.file_num).zfill(2) + '.npy')
        hf = h5py.File(file_dir, 'w')
        
        data_num = len(self.scenarios)
        total_frames = 0  
        for itr in range(data_num):
            total_frames += len(self.scenarios[itr]['frames'])

        frame_data = hf.create_dataset('frame_data', shape = (total_frames,), 
                                       dtype = np.float32)       
        x_data = hf.create_dataset('x_data', shape = (total_frames,), 
                                   dtype = np.float32)       
        y_data = hf.create_dataset('y_data', shape = (total_frames,), 
                                   dtype = np.float32)       
        
        file_ids = hf.create_dataset('file_ids', shape = (total_frames,), 
                                     dtype = int)
        tv_data = hf.create_dataset('tv_data', shape = (total_frames,), 
                                    dtype = int)
        labels = hf.create_dataset('labels', shape = (total_frames,), 
                                   dtype = np.float32)
        state_povl_data = hf.create_dataset('state_povl', 
                                            shape = (total_frames, 27), 
                                            dtype = np.float32)
        state_wirth_data = hf.create_dataset('state_wirth', 
                                            shape = (total_frames, 18), 
                                            dtype = np.float32)
        
        state_constantx_data = hf.create_dataset('state_constantx_data', 
                                                 shape = (total_frames, 4), 
                                                 dtype = np.float32)
        output_states_data = hf.create_dataset('output_states_data', 
                                               shape = (total_frames, 2), 
                                               dtype = np.float32)
        
        cur_frame = 0
        #pdb.set_trace()
        for itr in range(data_num):
            scenario_length = len(self.scenarios[itr]['frames'])
            state_povl_data[cur_frame:(cur_frame+scenario_length), :] =\
                  self.scenarios[itr]['states_povl']
            state_wirth_data[cur_frame:(cur_frame+scenario_length), :] =\
                  self.scenarios[itr]['states_wirth']
            state_constantx_data[cur_frame:(cur_frame+scenario_length), :] = \
                self.scenarios[itr]['states_constantx']
            
            output_states_data[cur_frame:(cur_frame+scenario_length), :] = \
                self.scenarios[itr]['output_states']
            frame_data[cur_frame:(cur_frame+scenario_length)] = \
                self.scenarios[itr]['frames']
            x_data[cur_frame:(cur_frame+scenario_length)] = \
                self.scenarios[itr]['x']
            y_data[cur_frame:(cur_frame+scenario_length)] = \
                self.scenarios[itr]['y']
            tv_data[cur_frame:(cur_frame+scenario_length)] = \
                self.scenarios[itr]['tv']
            file_ids[cur_frame:(cur_frame+scenario_length)] = \
                self.scenarios[itr]['file']
            labels[cur_frame:(cur_frame+scenario_length)] = \
                self.scenarios[itr]['label']
            cur_frame +=scenario_length
            
        assert(cur_frame == total_frames)
        hf.close()
        np.save(npy_dir, total_frames) 
    
    def render_scenarios(self)-> "Number of rendered and saved scenarios":
        saved_data_number = 0
        
        for scenario_idx, scenario in enumerate(self.scenarios):        
            if scenario_idx%500 == 0:
                print('Scenario {} out of: {}'\
                      .format(scenario_idx, len(self.scenarios)))
            tv_id = scenario['tv']
            img_frames = []
            states_povl = []
            states_constantx = []
            states_wirth = []
            output_states = []
            number_of_fr = len(scenario['frames'])
            tv_lane_ind = None   
            for fr in range(number_of_fr):
                frame = scenario['frames'][fr]
                img_frames.append(frame)
                
                svs_ids = scenario['svs']['id'][:,fr]
                state_povl, state_constantx, state_wirth, output_state, tv_lane_ind =\
                      self.calc_states_povl(
                    self.frames_data[self.frame_list.index(frame)],
                    tv_id, 
                    svs_ids,
                    frame,
                    tv_lane_ind
                    )
                output_states.append(output_state)
                # first time-step is repeated so that when we calc the diff, 
                # the first timestep would be zero displacement.
                if fr==0:  
                    output_states.append(output_state)
                
                states_povl.append(state_povl)
                states_constantx.append(state_constantx)
                states_wirth.append(state_wirth)
                    
            self.scenarios[scenario_idx]['states_povl'] = np.array(states_povl)
            self.scenarios[scenario_idx]['states_constantx'] = \
                np.array(states_constantx)
            self.scenarios[scenario_idx]['states_wirth'] = \
                np.array(states_wirth)
            output_states = np.array(output_states)
            output_states = output_states[1:,:]- output_states[:-1,:] 
            self.scenarios[scenario_idx]['output_states'] = output_states
            saved_data_number += 1
            
        return saved_data_number

    def calc_states_povl(
        self, 
        frame_data:'Data array of current frame', 
        tv_id:'ID of the TV', 
        svs_ids:'IDs of the SVs', 
        frame:'frame',
        tv_lane_ind:'TV lane index'):
        
        assert(frame_data[rc.FRAME][0]==frame)   
        assert(tv_id in frame_data[rc.TRACK_ID])
        if p.SVS_FORMAT != 'povl':
            raise(ValueError('This functions requires povl SV format'))
        
        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
        
        # exid version
        lateral_pos = lambda itr: frame_data[rc.Y2LANE][itr]
        
        rel_distance_x = lambda itr: abs(frame_data[rc.S][itr] \
                                         - frame_data[rc.S][tv_itr])
        rel_distance_y = lambda itr: abs(frame_data[rc.D][itr] \
                                         - frame_data[rc.D][tv_itr])
        
        # TV lane markings and lane index
        if p.DATASET == 'Processed_highD':
            tv_lane_markings = self.metas[rc.LOWER_LANE_MARKINGS]
            tv_lane_ind = frame_data[rc.LANE_ID][tv_itr]-len(self.metas[rc.UPPER_LANE_MARKINGS])-2
            
            tv_lane_ind = int(tv_lane_ind)
            tv_left_lane_ind =  tv_lane_ind
            if tv_lane_ind+1 >=len(tv_lane_markings) or tv_lane_ind<0:
                raise(ValueError('Unexpected target vehicle lane'))
            lane_width = (tv_lane_markings[tv_lane_ind+1]-tv_lane_markings[tv_lane_ind])
            #print('lane width: {}'.format(lane_width))
        else:
            tv_lane_ind = frame_data[rc.LANE_ID][tv_itr]
            lane_width = frame_data[rc.LANE_WIDTH][tv_itr]
        ## Output States:
        output_state = np.zeros((2))
        output_state[0] = frame_data[rc.D][tv_itr]
        output_state[1] = frame_data[rc.S][tv_itr]
        
        svs_itr = np.array([np.nonzero(frame_data[rc.TRACK_ID] == sv_id)[0][0]\
                             if sv_id!=0 and sv_id!=-1 else None \
                                for sv_id in svs_ids])
        # svs : [pv_id, fv_id, rv1_id, rv2_id, rv3_id, lv1_id, lv2_id, lv3_id]
        pv_itr = svs_itr[0]
        fv_itr = svs_itr[1]
        lcp_itr = svs_itr[2]
        lfp_itr = svs_itr[3]
        lcf_itr = svs_itr[4]
        lff_itr = svs_itr[5]
        rcp_itr = svs_itr[6]
        rfp_itr = svs_itr[7]
        rcf_itr = svs_itr[8]
        rff_itr = svs_itr[9]
        
        lateral_pos_highd =  lambda itr, lane_itr: frame_data[rc.D][itr] - tv_lane_markings[lane_itr]
        ######################## State ConstantX ############################
        state_constantx = np.zeros((4))
        state_constantx[0] = frame_data[rc.D_VELOCITY][tv_itr] 
        state_constantx[1] = frame_data[rc.S_VELOCITY][tv_itr] 
        state_constantx[2] = frame_data[rc.D_ACCELERATION][tv_itr] 
        state_constantx[3] = frame_data[rc.S_ACCELERATION][tv_itr] 
        
        ########################## State POVL ################################
        state_povl = np.zeros((27)) # a proposed features  
        # (1) Lateral Pos
        if p.DATASET=='Processed_highD':
            state_povl[0] = lateral_pos_highd(tv_itr, tv_left_lane_ind)
        else:
            state_povl[0] = lateral_pos(tv_itr)
        # (2) Long Velo
        state_povl[1] = frame_data[rc.S_VELOCITY][tv_itr]
        # (3)Lat ACC
        state_povl[2] = frame_data[rc.D_ACCELERATION][tv_itr]
        # (4)Long ACC
        state_povl[3] = frame_data[rc.S_ACCELERATION][tv_itr]
        # (5) PV X
        state_povl[4] = rel_distance_x(pv_itr) if pv_itr != None else 400
        # (6) PV Y
        state_povl[5] = rel_distance_y(pv_itr) if pv_itr != None else 0
        # (7) FV X
        state_povl[6] = rel_distance_x(fv_itr) if fv_itr != None else 400
        # (8) FV Y
        state_povl[7] = rel_distance_y(pv_itr) if pv_itr != None else 0
        
        # (9) RCP X
        state_povl[8] = rel_distance_x(rcp_itr) if rcp_itr != None else 400
        # (10) RCP Y
        state_povl[9] = rel_distance_y(rcp_itr) if rcp_itr != None else 30
        
        # (11) RFP X
        state_povl[10] = rel_distance_x(rfp_itr) if rfp_itr != None else 400
        # (12) RFP Y
        state_povl[11] = rel_distance_y(rfp_itr) if rfp_itr != None else 30
        
        # (13) RCF X
        state_povl[12] = rel_distance_x(rcf_itr) if rcf_itr != None else 400
        # (14) RCF Y
        state_povl[13] = rel_distance_y(rcf_itr) if rcf_itr != None else 30
        
        # (15) RFF X
        state_povl[14] = rel_distance_x(rff_itr) if rff_itr != None else 400
        # (16) RFF Y
        state_povl[15] = rel_distance_y(rff_itr) if rff_itr != None else 30
        
        # (17) LCP X
        state_povl[16] = rel_distance_x(lcp_itr) if lcp_itr != None else 400
        # (18) LCP Y
        state_povl[17] = rel_distance_y(lcp_itr) if lcp_itr != None else 30
        
        # (19) LFP X
        state_povl[18] = rel_distance_x(lfp_itr) if lfp_itr != None else 400
        # (20) LFP Y
        state_povl[19] = rel_distance_y(lfp_itr) if lfp_itr != None else 30
        
        # (21) LCF X
        state_povl[20] = rel_distance_x(lcf_itr) if lcf_itr != None else 400
        # (22) LCF Y
        state_povl[21] = rel_distance_y(lcf_itr) if lcf_itr != None else 30
        
        # (23) LFF X
        state_povl[22] = rel_distance_x(lff_itr) if lff_itr != None else 400
        # (24) LFF Y
        state_povl[23] = rel_distance_y(lff_itr) if lff_itr != None else 30
        
        # (25) Lane width
        state_povl[24] = lane_width
        if p.DATASET == 'Processed_highD':
            n_lane = len(tv_lane_markings)-1
            # (26) Right Lane Type # 0:normal, 1: expect merging 2:merge, 3:no lane  
            state_povl[25] = rf.get_lane_type_highd(tv_lane_ind+1, n_lane)
            # (27) Left Lane Type
            state_povl[26] = rf.get_lane_type_highd(tv_lane_ind-1, n_lane)
        else:
            n_lane = p.merge_lane_id[p.ind_list.index(self.file_num)]
            # (26) Right Lane Type # 0:normal, 1: expect merging 2:merge, 3:no lane  
            state_povl[25] = rf.get_lane_type(tv_lane_ind+1, n_lane)
            # (27) Left Lane Type
            state_povl[26] = rf.get_lane_type(tv_lane_ind-1, n_lane)



        ######################### State Merging ##############################
        rel_velo_x = lambda itr: frame_data[rc.S_VELOCITY][itr] - frame_data[rc.S_VELOCITY][tv_itr] #transform from [-1,1] to [0,1]
        rel_velo_y = lambda itr: frame_data[rc.D_VELOCITY][itr] - frame_data[rc.D_VELOCITY][tv_itr] #transform from [-1,1] to [0,1]
        rel_acc_x = lambda itr: frame_data[rc.S_ACCELERATION][itr] - frame_data[rc.S_ACCELERATION][tv_itr]
        
        state_wirth = np.zeros((18))
        
        if p.DATASET == 'Processed_highD':
            n_lane = len(tv_lane_markings)-1
            # (0) Right Lane Type # 0:normal, 1: expect merging 2:merge, 3:no lane  
            state_wirth[0] = rf.get_lane_type_highd(tv_lane_ind+1, n_lane)
            # (1) Left Lane Type
            state_wirth[1] = rf.get_lane_type_highd(tv_lane_ind-1, n_lane)
        else:
            n_lane = p.merge_lane_id[p.ind_list.index(self.file_num)]
            # (0) Right Lane Type # 0:normal, 1: expect merging 2:merge, 3:no lane  
            state_wirth[0] = rf.get_lane_type(tv_lane_ind+1, n_lane)
            # (1) Left Lane Type
            state_wirth[1] = rf.get_lane_type(tv_lane_ind-1, n_lane)
        # (3) lane width,  
        state_wirth[2] = lane_width 
        # (4) Longitudinal distance of TV to PV, 
        state_wirth[3] = rel_distance_x(pv_itr) if pv_itr != None else 400 
        # (5)Longitudinal distance of TV to RPV, 
        state_wirth[4] = rel_distance_x(rcp_itr) if rcp_itr != None else 400  
        # (6)Longitudinal distance of TV to FV, 
        state_wirth[5] = rel_distance_x(fv_itr) if fv_itr != None else 400 
        # (7)lateral distance of TV to the left lane marking, 
        if p.DATASET=='Processed_highD':
            state_wirth[6] = lateral_pos_highd(tv_itr, tv_left_lane_ind)
        else:
            state_wirth[6] = lateral_pos(tv_itr)
        # (8)lateral distance of TV to RV, 
        state_wirth[7] = rel_distance_y(rcf_itr) if rcf_itr != None else 3*lane_width 
        # (9)lateral distance of TV to RFV, 
        state_wirth[8] = rel_distance_y(rff_itr) if rff_itr != None else 3*lane_width 
        # (10) relative longitudinal velocity of TV w.r.t. PV,
        state_wirth[9] = rel_velo_x(pv_itr) if pv_itr != None else 0  
        # (11) relative longitudinal velocity of TV w.r.t. FV 
        state_wirth[10] = rel_velo_x(fv_itr) if fv_itr != None else 0
        # (12)Relative lateral velocity of TV w.r.t. PV, 
        state_wirth[11] = rel_velo_y(pv_itr) if pv_itr != None else 0
        # (13)Relative lateral velocity of TV w.r.t. RPV,
        state_wirth[12] = rel_velo_y(rcp_itr) if rcp_itr != None else 0  
        # (14)Relative lateral velocity of TV w.r.t. RV, 
        state_wirth[13] = rel_velo_y(rcf_itr) if rcf_itr != None else 0
        # (15)Relative lateral velocity of TV w.r.t. LV, 
        state_wirth[14] = rel_velo_y(lcf_itr) if lcf_itr != None else 0
        # (16) longitudinal acceleration of the TV, 
        state_wirth[15] = frame_data[rc.S_ACCELERATION][tv_itr]
        # (17) relative longitudinal acceleration of the TV w.r.t RPV, 
        state_wirth[16] = rel_acc_x(rcp_itr) if rcp_itr != None else 0
        # (18) lateral acceleration of the prediction target
        state_wirth[17] = frame_data[rc.D_ACCELERATION][tv_itr]

        return state_povl, state_constantx, state_wirth, output_state, tv_lane_ind, 


    def calc_states_mmntp(
        self, 
        frame_data:'Data array of current frame', 
        tv_id:'ID of the TV', 
        svs_ids:'IDs of the SVs', 
        frame:'frame',
        tv_lane_ind:'TV lane index'):
        
        assert(frame_data[rc.FRAME][0]==frame)   
        if p.SVS_FORMAT != 'highD':
            raise(ValueError('This functions requires highD SV format'))
        tv_itr = np.nonzero(frame_data[rc.TRACK_ID] == tv_id)[0][0]
        
        

        
        # exid version
        lateral_pos = lambda itr: frame_data[rc.Y2LANE][itr]
        
        rel_distance_x = lambda itr: (frame_data[rc.X][itr] - frame_data[rc.X][tv_itr])
        rel_distance_y = lambda itr: (frame_data[rc.Y][itr] - frame_data[rc.Y][tv_itr])
        
        # TV lane markings and lane index
        '''
        tv_lane_markings = (self.metas[rc.UPPER_LANE_MARKINGS]) if driving_dir\
              == 1 else (self.metas[rc.LOWER_LANE_MARKINGS])
        
        
        if driving_dir ==1:
            tv_lane_ind = frame_data[rc.LANE_ID][tv_itr]-2
        else:
            tv_lane_ind = frame_data[rc.LANE_ID][tv_itr]-\
                len(self.metas[rc.UPPER_LANE_MARKINGS])-2
        
        tv_lane_ind = int(tv_lane_ind)
        tv_left_lane_ind = tv_lane_ind + 1 if driving_dir==1 else tv_lane_ind
        if tv_lane_ind+1 >=len(tv_lane_markings):
            return True, 0, 0, 0, 0, 0, 0
        lane_width = (tv_lane_markings[tv_lane_ind+1]-tv_lane_markings[tv_lane_ind])
        #print('lane width: {}'.format(lane_width))
       '''
        tv_lane_ind = frame_data[rc.LANE_ID][tv_itr]-2
        lane_width = frame_data[rc.LANE_WIDTH][tv_itr]
        ## Output States:
        output_state = np.zeros((2))
        output_state[0] = frame_data[rc.Y][tv_itr]
        output_state[1] = frame_data[rc.X][tv_itr]
        
        
        svs_itr = np.array([np.nonzero(frame_data[rc.TRACK_ID] == sv_id)[0][0]\
                             if sv_id!=0 and sv_id!=-1 else None for sv_id in svs_ids])
        # svs : [pv_id, fv_id, rv1_id, rv2_id, rv3_id, lv1_id, lv2_id, lv3_id]
        pv_itr = svs_itr[0]
        fv_itr = svs_itr[1]
        rv1_itr = svs_itr[2]
        rv2_itr = svs_itr[3]
        rv3_itr = svs_itr[4]
        lv1_itr = svs_itr[5]
        lv2_itr = svs_itr[6]
        lv3_itr = svs_itr[7]
        
        
        ######################### State Merging ##############################
        state_merging = np.zeros((21)) # a proposed features  
        # (1) Lateral Pos
        state_merging[0] = lateral_pos(tv_itr)
        # (2) Long Velo
        state_merging[1] = frame_data[rc.X_VELOCITY][tv_itr]
        # (3)Lat ACC
        state_merging[2] = frame_data[rc.Y_ACCELERATION][tv_itr]
        # (4)Long ACC
        state_merging[3] = frame_data[rc.X_ACCELERATION][tv_itr]
        # (5) PV X
        state_merging[4] = rel_distance_x(pv_itr) if pv_itr != None else 400
        # (6) PV Y
        state_merging[5] = rel_distance_y(pv_itr) if pv_itr != None else 30
        # (7) FV X
        state_merging[6] = rel_distance_x(fv_itr) if fv_itr != None else 400
        # (8) FV Y
        state_merging[7] = rel_distance_y(pv_itr) if pv_itr != None else 30
        # (9) RV1 X
        state_merging[8] = rel_distance_x(rv1_itr) if rv1_itr != None else 400
        # (10) RV1 Y
        state_merging[9] = rel_distance_y(pv_itr) if pv_itr != None else 30
        # (11) RV2 X
        state_merging[10] = rel_distance_x(rv2_itr) if rv2_itr != None else 400
        # (12) RV2 Y
        state_merging[11] = rel_distance_y(pv_itr) if pv_itr != None else 30
        # (13) RV3 X
        state_merging[12] = rel_distance_x(rv3_itr) if rv3_itr != None else 400
        # (14) RV3 Y
        state_merging[13] = rel_distance_y(rv3_itr) if rv3_itr != None else 30
        # (15) LV1 X
        state_merging[14] = rel_distance_x(lv1_itr) if lv1_itr != None else 400
        # (16) LV1 Y
        state_merging[15] = rel_distance_y(lv1_itr) if lv1_itr != None else -30
        # (17) LV2 X
        state_merging[16] = rel_distance_x(lv2_itr) if lv2_itr != None else 400
        # (18) LV2 Y
        state_merging[17] = rel_distance_y(lv2_itr) if lv2_itr != None else -30
        # (19) LV3 X
        state_merging[18] = rel_distance_x(lv3_itr) if lv3_itr != None else 400
        # (20) LV3 Y
        state_merging[19] = rel_distance_y(lv3_itr) if lv3_itr != None else -30
        # (21) Lane width
        state_merging[20] = lane_width

        return state_merging, output_state, tv_lane_ind, 
    
    def update_dirs(self):
        '''
        self.LC_cropped_imgs_dir = self.LC_cropped_imgs_rdir
        if not os.path.exists(self.LC_cropped_imgs_dir):
            os.makedirs(self.LC_cropped_imgs_dir)
        
        for i in range(3):
            label_dir = os.path.join(self.LC_cropped_imgs_dir, str(i))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir) 

        
        self.LC_whole_imgs_dir = self.LC_whole_imgs_rdir
        if not os.path.exists(self.LC_whole_imgs_dir):
            os.makedirs(self.LC_whole_imgs_dir)
    
        for i in range(3):
            label_dir = os.path.join(self.LC_whole_imgs_dir, str(i))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir) 
        '''
        self.LC_image_dataset_dir = self.LC_image_dataset_rdir
        if not os.path.exists(self.LC_image_dataset_dir):
            os.makedirs(self.LC_image_dataset_dir)
        
        
    
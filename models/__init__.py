import logging
from models.vmg import VMG
logger = logging.getLogger('base')


def create_model(config):
    network_config = config['network']

    if config['model'] == 'VMG':
        image_size = [int(config['dataset']['image_shape_r'][1]/config['scale']), int(config['dataset']['image_shape_r'][2]/config['scale'])]
        model = VMG(
            embed_dim=network_config['embed_dim'],
            depths=network_config['depths'],
            mlp_ratio=network_config['mlp_ratio'], n_groups=network_config['n_groups'],
            num_heads=network_config['num_heads'],
            window_sizes=network_config['window_sizes'],
            num_frames=network_config['num_frames'],
            back_RBs=network_config['back_RBs'],
            spynet_pretrained=network_config['spynet'],
            image_size=image_size,
            is_train=config['is_train'],
            if_print=network_config['if_print'],
            ltam=network_config['ltam'],
            traj_win=network_config['traj_win'], traj_keyframes_n=network_config['traj_keyframes_n'], traj_heads=network_config['traj_heads'],
            temporal_type=network_config['temporal_type'], temporal_empty=network_config['temporal_empty'],
            traj_res_n=network_config['traj_res_n'],
            deform_groups=network_config['deform_groups'], max_residual_scale=network_config['max_res_scale'],
            spatial_type=network_config['spatial_type'],
            mdsc=network_config['use_mdsc'], if_concat=network_config['if_concat'],
            flow_smooth=network_config['flow_smooth'], smooth_region_range=network_config['smooth_region_range'],
            retention_decay=network_config['ret_decay'],
            non_linear=network_config['non_linear'],
            gating=network_config['gating'], symm=network_config['if_symm'], symm_act=network_config['symm_act'],
            relu_scale=network_config['relu_scale'], relu_scale_norm=network_config['relu_scale_norm'],
            ffn_type=network_config['ffn_type'],
            mixer_type=network_config['mixer_type'], mixer_n=network_config['mixer_n'],
            r_scaling=network_config['r_scaling'],
            chunk_ratios=network_config['chunk_ratios'],
            traj_mode=network_config['traj_mode'], twins=network_config['twins'], traj_scale=network_config['traj_scale'], traj_refine=network_config['traj_refine'],
            m_scaling=network_config['m_scaling'],
            if_local_fuse=network_config['if_local_fuse'],
            channel_mixer=network_config['channel_mixer']
        )   
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(config['model']))

    # logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))

    return model

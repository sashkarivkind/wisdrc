import argparse

def parse_commandline(return_parser=False):
    parser = argparse.ArgumentParser()

    #general parameters
    parser.add_argument('--run_name_prefix', default='noname', type=str, help='')
    parser.add_argument('--local_save_path', default='/home/bnapp/arivkindNet/drc_saves/', type=str, help='path to local save directory (used outside lsf ')
    parser.add_argument('--lsb_save_path', default='/home/labs/ahissarlab/arivkind/drc_saves/', type=str, help='path to wexac save directory')
    parser.add_argument('--run_index', default=10, type=int, help='run_index')
    parser.add_argument('--verbose', default=2, type=int, help='run_index')

    parser.add_argument('--student_args', default=None, type=str, help='(auxillary) filename of pickle file with student arguments')

    parser.add_argument('--n_classes', default=10, type=int, help='classes')

    parser.add_argument('--testmode', dest='testmode', action='store_true')
    parser.add_argument('--no-testmode', dest='testmode', action='store_false')

    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--gpu_id', default=-1, type=int, help='gpu id to use, -1 for default')
    parser.add_argument('--image_h', default=224, type=int, help='high res frame HxW')
    parser.add_argument('--image_w', default=224, type=int,help='high res frame HxW')


    ### student parameters

    parser.add_argument('--stu_steps_per_epoch', default=1000, type=int, help='batches per epoch, student pre-training')
    parser.add_argument('--fine_tuning_steps_per_epoch', default=1000, type=int, help='batches per epoch, fine tuning')
    parser.add_argument('--epochs', default=1, type=int, help='num training epochs')
    parser.add_argument('--int_epochs', default=1, type=int, help='num internal training epochs')
    parser.add_argument('--decoder_epochs', default=40, type=int, help='num internal training epochs')
    parser.add_argument('--num_features', default=64, type=int, help='number of features at the DRC interface')
    parser.add_argument('--rnn_layer1', default=32, type=int, help='legacy to be discarded')
    parser.add_argument('--rnn_layer2', default=64, type=int, help='legacy to be discarded')
    parser.add_argument('--time_pool', default=0, help='time dimension pooling to use - max_pool, average_pool, 0')

    parser.add_argument('--student_block_size', default=1, type=int, help='number of repetition of each convlstm block')
    parser.add_argument('--upsample', default=0, type=int, help='spatial upsampling of input 0 for no')
    parser.add_argument('--amp', default=4, type=int, help='amplitude of trajectory')

    parser.add_argument('--centered_offsets', dest='centered_offsets', action='store_true')
    parser.add_argument('--no-centered_offsets', dest='centered_offsets', action='store_false')

    parser.add_argument('--nets_to_eval', default=None, type=str, help='text file with a list of nets to eval by eval_loops.py')

    parser.add_argument('--conv_rnn_type', default='lstm', type=str, help='conv_rnn_type')
    parser.add_argument('--student_nl', default='relu', type=str, help='non linearity')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout1')
    parser.add_argument('--rnn_dropout', default=0.0, type=float, help='dropout1')
    parser.add_argument('--teacher_net_initial_weight', default=0.9, type=float, help='teacher_net_initial_weight')
    parser.add_argument('--reference_feature_stats', default=None, type=str, help='pickle file with reference feature statistics')
    parser.add_argument('--loss_coeffs',
                        type=float, action='append',
                        dest='loss_coeffs',
                        default=[],
                        help='list of coefficients for complex loss')
    parser.add_argument('--pretrained_student_path', default=None, type=str, help='pretrained student, works only with student3')
    parser.add_argument('--pretrained_student_model', default=None, type=str, help='pretrained student model')
    parser.add_argument('--pretrained_decoder_path', default=None, type=str, help='pretrained decoder, UNDER CONSTRUCTION')
    parser.add_argument('--out_weight_pickle_file', default=None, type=str, help='pickle to put all the wights')

    parser.add_argument('--pos_det', default=None, type=str, help='positoin detector model')
    parser.add_argument('--pd_n_layers', default=3, type=int, help='amplitude of trajectory')
    parser.add_argument('--pd_n_units', default=32, type=int, help='amplitude of trajectory')
    parser.add_argument('--pd_d_filter', default=3, type=int, help='amplitude of trajectory')

    parser.add_argument('--decoder_optimizer', default='Adam', type=str, help='Adam or SGD')

    parser.add_argument('--preprocessing', default='keras_resnet50', type=str, help='low-level preprocessing')
    parser.add_argument('--teacher_model', default='keras_resnet50', type=str, help='low-level preprocessing')
    parser.add_argument('--split_after_layer', default=None, type=str, help='low-level preprocessing')


    parser.add_argument('--skip_student_training', dest='skip_student_training', action='store_true')
    parser.add_argument('--no-skip_student_training', dest='skip_student_training', action='store_false')

    parser.add_argument('--fine_tune_student', dest='fine_tune_student', action='store_true')
    parser.add_argument('--no-fine_tune_student', dest='fine_tune_student', action='store_false')

    parser.add_argument('--layer_norm_student', dest='layer_norm_student', action='store_true')
    parser.add_argument('--no-layer_norm_student', dest='layer_norm_student', action='store_false')

    parser.add_argument('--batch_norm_student', dest='batch_norm_student', action='store_true')
    parser.add_argument('--no-batch_norm_student', dest='batch_norm_student', action='store_false')

    parser.add_argument('--val_set_mult', default=5, type=int, help='repetitions of validation dataset to reduce trajectory noise')

    parser.add_argument('--use_teacher_net_at_low_res', dest='use_teacher_net_at_low_res' , action='store_true')
    parser.add_argument('--no-use_teacher_net_at_low_res', dest='use_teacher_net_at_low_res', action='store_false')

    parser.add_argument('--teacher_only_at_low_res', dest='teacher_only_at_low_res', action='store_true')
    parser.add_argument('--no-teacher_only_at_low_res', dest='teacher_only_at_low_res', action='store_false')

    parser.add_argument('--baseline_rggb_mode', dest='baseline_rggb_mode', action='store_true')
    parser.add_argument('--no-baseline_rggb_mode', dest='baseline_rggb_mode', action='store_false')



    ### syclop parameters
    parser.add_argument('--trajectory_index', default=0, type=int, help='trajectory index - set to 0 because we use multiple trajectories')
    parser.add_argument('--n_samples', default=5, type=int, help='n_samples')
    parser.add_argument('--res', default=8, type=int, help='resolution')
    parser.add_argument('--central_squeeze_and_pad_factor', default=-1, type=float, help='factor by which to squeeze the original image, before applying low resolution sensor ')

    parser.add_argument('--manual_trajectories', dest='manual_trajectories', action='store_true')
    parser.add_argument('--no-manual_trajectories', dest='manual_trajectories', action='store_false')

    parser.add_argument('--trajectory_file', default=None, type=str, help='numpy file to take trajectories from')
    parser.add_argument('--trajectories_num', default=-1, type=int, help='number of trajectories to use')
    parser.add_argument('--broadcast', default=0, type=int, help='1-integrate the coordinates by broadcasting them as extra dimentions, 2- add coordinates as an extra input')
    parser.add_argument('--style', default='spiral_2dir2', type=str, help='choose syclops style of motion')
    parser.add_argument('--loss', default='mean_squared_error', type=str, help='loss type for student')
    parser.add_argument('--noise', default=0.15, type=float, help='added noise to the const_p_noise style')
    parser.add_argument('--max_length', default=5, type=int, help='choose syclops max trajectory length')
    parser.add_argument('--rggb_ext_type', default=1, type=int, help='how to extend from 28x28 to 56x56, 1 - at input, 2-upsample after convrnn 3-deconv after convrnn')
    parser.add_argument('--kernel_size', default=3, type=int, help='how to extend from 28x28 to 56x56, 1 - at input, 2-upsample after convrnn 3-deconv after convrnn')

    parser.add_argument('--random_n_samples', default=0, type=int, help='weather to drew random lengths of trajectories')

    ### teacher network parameters
    parser.add_argument('--dataset_dir', default='/home/bnapp/datasets/tensorflow_datasets/imagenet2012/5.0.0/', type=str, help='path to the dataset')

    parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
    parser.add_argument('--student_version', default=3, type=int, help='student version')

    parser.add_argument('--last_layer_size', default=128, type=int, help='last_layer_size')


    parser.add_argument('--dropout1', default=0.2, type=float, help='dropout1')
    parser.add_argument('--dropout2', default=0.0, type=float, help='dropout2')
    parser.add_argument('--dataset_norm', default=128.0, type=float, help='dropout2')
    parser.add_argument('--syclopic_norm', default=256.0, type=float, help='redundant legacy normalization')
    parser.add_argument('--traj_scaling', default=1.0, type=float, help='scaling for the trajectory data')
    parser.add_argument('--traj_bias', default=0.0, type=float, help='bias for the trajectory data')
    parser.add_argument('--dataset_center', dest='dataset_center', action='store_true')
    parser.add_argument('--no-dataset_center', dest='dataset_center', action='store_false')

    parser.add_argument('--dense_interface', dest='dense_interface', action='store_true')
    parser.add_argument('--no-dense_interface', dest='dense_interface', action='store_false')

    parser.add_argument('--layer_norm_res', dest='layer_norm_res', action='store_true')
    parser.add_argument('--no-layer_norm_res', dest='layer_norm_res', action='store_false')

    parser.add_argument('--layer_norm_2', dest='layer_norm_2', action='store_true')
    parser.add_argument('--no-layer_norm_2', dest='layer_norm_2', action='store_false')

    parser.add_argument('--skip_conn', dest='skip_conn', action='store_true')
    parser.add_argument('--no-skip_conn', dest='skip_conn', action='store_false')

    parser.add_argument('--last_maxpool_en', dest='last_maxpool_en', action='store_true')
    parser.add_argument('--no-last_maxpool_en', dest='last_maxpool_en', action='store_false')

    parser.add_argument('--resnet_mode', dest='resnet_mode', action='store_true')
    parser.add_argument('--no-resnet_mode', dest='resnet_mode', action='store_false')

    parser.add_argument('--nl', default='relu', type=str, help='non linearity')

    parser.add_argument('--stopping_patience', default=10, type=int, help='stopping patience')
    parser.add_argument('--learning_patience', default=5, type=int, help='stopping patience')
    parser.add_argument('--manual_suffix', default='', type=str, help='manual suffix')

    parser.add_argument('--shuffle_traj', dest='shuffle_traj', action='store_true')
    parser.add_argument('--no-shuffle_traj', dest='shuffle_traj', action='store_false')

    parser.add_argument('--tf_down_sample', dest='tf_down_sample', action='store_true')
    parser.add_argument('--no-tf_down_sample', dest='tf_down_sample', action='store_false')

    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
    parser.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false')

    parser.add_argument('--rotation_range', default=0.0, type=float, help='dropout1')
    parser.add_argument('--width_shift_range', default=0.1, type=float, help='dropout2')
    parser.add_argument('--height_shift_range', default=0.1, type=float, help='dropout2')

    ##advanced trajectory parameters
    parser.add_argument('--time_sec', default=0.3, type=float, help='time for realistic trajectory')
    parser.add_argument('--traj_out_scale', default=4.0, type=float, help='scaling to match receptor size')

    parser.add_argument('--snellen', dest='snellen', action='store_true')
    parser.add_argument('--no-snellen', dest='snellen', action='store_false')

    parser.add_argument('--vm_kappa', default=0., type=float, help='factor for emulating sub and super diffusion')

    parser.add_argument('--unprocess_high_res', dest='unprocess_high_res', action='store_true')
    parser.add_argument('--no-unprocess_high_res', dest='unprocess_high_res', action='store_false')

    parser.add_argument('--enable_random_gains', dest='enable_random_gains', action='store_true')
    parser.add_argument('--no-enable_random_gains', dest='enable_random_gains', action='store_false')

    parser.add_argument('--enforce_zero_initial_offset_fea', dest='enforce_zero_initial_offset_fea', action='store_true')
    parser.add_argument('--no-enforce_zero_initial_offset_fea', dest='enforce_zero_initial_offset_fea', action='store_false')

    parser.add_argument('--enforce_zero_initial_offset_cls', dest='enforce_zero_initial_offset_cls', action='store_true')
    parser.add_argument('--no-enforce_zero_initial_offset_cls', dest='enforce_zero_initial_offset_cls', action='store_false')

    parser.add_argument('--rggb_mode', dest='rggb_mode', action='store_true')
    parser.add_argument('--no-rggb_mode', dest='rggb_mode', action='store_false')

    parser.add_argument('--skip_final_saves', dest='skip_final_saves', action='store_true')
    parser.add_argument('--no-skip_final_saves', dest='skip_final_saves', action='store_false')

    parser.add_argument('--evaluate_final_model', dest='evaluate_final_model', action='store_true')
    parser.add_argument('--no-evaluate_final_model', dest='evaluate_final_model', action='store_false')

    parser.add_argument('--varying_max_amp', dest='evaluate_final_model', action='store_true')
    parser.add_argument('--no-varying_max_amp', dest='evaluate_final_model', action='store_false')


    parser.set_defaults(data_augmentation=True,
                        skip_final_saves=False,
                        layer_norm_res=True,
                        layer_norm_student=True,
                        batch_norm_student=False,
                        layer_norm_2=True,
                        skip_conn=True,
                        last_maxpool_en=True,
                        testmode=False,
                        dataset_center=True,
                        dense_interface=False,
                        resnet_mode=False,
                        skip_student_training=False,
                        fine_tune_student=False,
                        snellen=True,
                        shuffle_traj=False,
                        tf_down_sample=False,
                        rggb_mode=True,
                        centered_offsets=False,
                        manual_trajectories = False,
                        unprocess_high_res = True,
                        enable_random_gains = True,
                        enforce_zero_initial_offset_fea=False,
                        enforce_zero_initial_offset_cls=False,
                        evaluate_final_model = False,
                        use_teacher_net_at_low_res = False,
                        teacher_only_at_low_res = False,
                        baseline_rggb_mode=False,
                        varying_max_amp=False

        )
    if return_parser:
        return parser
    else:
        config = parser.parse_args()
        config = vars(config)
        return config

#bsub -q gpu-long -app nvidia-gpu -env LSB_CONTAINER_IMAGE=nvcr.io/nvidia/tensorflow:21.11-tf2-py3 -gpu num=2:j_exclusive=yes -R "affinity[thread*4] select[mem>132384] rusage[mem=132384]"  -R "select[hname!=dgxws01]"    -R "select[hname!=dgxws02]"    -R "select[hname!=agn04]" -R "select[hname!=agn05]"   -R "select[hname!=hgn02]"     -R "select[hname!=hgn03]"  -R "select[hname!=hgn07]"    -R "select[hname!=hgn08]"   -R "select[hname!=hgn09]"    -R "select[hname!=hgn10]"    -R "select[hname!=hgn11]"     -R "select[hname!=hgn12]" -o out1Imgnet.%J -e err1Imgnet.%J     python full_learning_imagenet303.py --student_nl relu --dropout 0.0 --rnn_dropout 0.0 --conv_rnn_type lstm --n_samples 5  --max_length 5   --epochs 5 --int_epochs 0 --student_block_size 5 --time_pool average_pool --student_version 3 --resnet_mode --decoder_optimizer SGD --val_set_mult 1 --res 56 --verbose 2 --broadcast 0 --dataset_dir /shareDB/imagenet201/5.0.0/ --centered_offsets --amp 12 --rggb_ext_type 3 --kernel_size 3 --stu_steps_per_epoch 10000 --pretrained_student_path saved_models/noname_j451069_t1655727450_feature/noname_j451069_t1655727450_feature_net_ckpt
# -R "select[hname!=hgn07]"    -R "select[hname!=hgn08]"   -R "select[hname!=hgn09]"    -R "select[hname!=hgn10]"    -R "select[hname!=hgn11]"     -R "select[hname!=hgn12]" -o out1Imgnet.%J -e err1Imgnet.%J     python full_learning_imagenet303.py --student_nl relu --dropout 0.0 --rnn_dropout 0.0 --conv_rnn_type lstm --n_samples 5  --max_length 5   --epochs 5 --int_epochs 0 --student_block_size 5 --time_pool average_pool --student_version 3 --resnet_mode --decoder_optimizer SGD --val_set_mult 1 --res 56 --verbose 2 --broadcast 0 --dataset_dir /shareDB/imagenet201/5.0.0/ --centered_offsets --amp 12 --rggb_ext_type 3 --kernel_size 3 --stu_steps_per_epoch 1000

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True
        
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of ADAM')
        self.parser.add_argument('--color_jitter', type=float, default=0.0, help='0.7 is only for RobotCar, 0.0 for 7Scenes')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the model specified by the "which_epoch" argument')
        self.parser.add_argument('--epochs', type=int, default=500)
        self.parser.add_argument('--lr', type=float, default=8e-5, help='initial learning rate for ADAM')
        self.parser.add_argument('--update_lr', type=bool, default=False, help='decide whether to update the learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=25, help='number of iterations starting to decay')
        self.parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
        self.parser.add_argument('--save_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--val_freq', type=int, default=10, help='frequency of showing evaluation results on console')
        self.parser.add_argument('--which_epoch', default=0, help='which epoch to load if continuing training')


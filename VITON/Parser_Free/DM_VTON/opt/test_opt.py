from .base_opt import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def _add_args(self) -> None:
        super()._add_args()            

        self.parser.add_argument('--save_img', default=True, action='store_true', help='save the tryon images')

        # Checkpoints
        self.parser.add_argument('--pf_warp_checkpoint', type=str, default='DM-VTON/checkpoints/dmvton_pf_warp.pt', help='load the pretrained model from the specified location')
        self.parser.add_argument('--pf_gen_checkpoint', type=str, default='DM-VTON/checkpoints/dmvton_pf_gen.pt', help='load the pretrained model from the specified location')

        # Setting
        self.parser.add_argument('--align_corners', default=True, action='store_true', help='align corners for grid_sample')
class LidarLstmDataset(udata.Dataset):
	def __init__(self, x, y):
        super(PieceWiseConstantDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
    	return x.shape[0]

    def __get_item__(self,index):
    	return x[index],y[index]

    # Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # outputs = {}
    # for i, imgs in enumerate(dataloader):
    #     imgs_black = Variable(imgs["black"].type(Tensor))
    #     imgs_black_orig = Variable(imgs["orig"].type(Tensor))
    #     gen_ab = model(imgs_black)
    #     gen_ab.detach_
    #     gen_color = postprocess_tens_new(imgs_black_orig, gen_ab)[0].transpose(1,2,0)
    #     outputs[imgs["path"][0]] = gen_color
    # return outputs
from build_market_model import *


def generate_rationale_list(
    dataset, market_model, market_gen,
    industry_model, industry_gen, args,
    cuda=False
):
    train_loader = utils.get_train_loader(dataset, args)
    thres = 0.0001
    index = 1

    tr_word_list = []
    tr_rat_list = []
    tr_dir = []
    mar_word_list = []
    mar_rat_list = []
    mar_dir = []
    ct = 0
    for loader in train_loader:
        x_indx = loader['inputs']
        y = loader['labels']
        input_mask = loader['input_mask']
        if cuda:
            x_indx, y, input_mask = (
                x_indx.cuda(), y.cuda(), input_mask.cuda()
            )

        mask = market_gen(x_indx, input_mask)
        output = market_model(x_indx, mask)
        output = output.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        mask_tech = industry_gen(x_indx, input_mask)
        output_tech = industry_model(x_indx, mask_tech)
        output_tech = output_tech.detach().cpu().numpy()
        mask_tech = mask_tech.detach().cpu().numpy()

        wds_inp = x_indx.cpu().numpy()
        x_indx = x_indx.detach().cpu()
        rat_indx = x_indx * mask[:, :, 0]
        tech_rat_indx = x_indx * mask_tech[:, :, 0]
        for btc_id in range(wds_inp.shape[0]):
            if (
                np.abs(output[btc_id, 0]) >= thres 
                and np.abs(output_tech[btc_id, 0]) >= thres
            ):
                ct += 1
                tr_word_list.append(
                    x_indx[btc_id,:].tolist()
                )
                tr_rat_list.append(
                    tech_rat_indx[btc_id,:].tolist()
                )
                mar_word_list.append(
                    x_indx[btc_id,:].tolist()
                )
                mar_rat_list.append(
                    rat_indx[btc_id,:].tolist()
                )
                if loader['labels'][btc_id] >= 0:
                    tr_dir.append(1)
                    mar_dir.append(1)
                else:
                    tr_dir.append(-1)
                    mar_dir.append(-1)
            
        if index % 100 == 0:
            print(index)
        index += 1

    return (
        tr_word_list,  tr_rat_list, tr_dir, 
        mar_word_list, mar_rat_list, mar_dir
    )


if __name__ == "__main__":
    # Load Stuff
    args = generic.init_args()
    args["model_path"] = "models/rnn_market_model.pt"
    args["gen_path"] = "models/rnn_gen_model.pt"
    with open("data/data.pkl", "rb") as file:
        data = pickle.load(file)
    init_date = pd.to_datetime("20140101", format="%Y%m%d")
    fin_date = pd.to_datetime("20160531", format="%Y%m%d")
    # 20160601 - 20161231 is left out for testing
    
    od, date_count = load_data(data, init_date, fin_date)
    sel_ret = load_market_data('data/ff5_daily_clean.csv')
    documents, ky_ret, ky_dates = prepare_data(data, sel_ret, date_count)
    word2embedding, embedding_dim = load_glove_embedding('data/glove.6B.100d.txt')
    preprocess_documents(documents, word2embedding)
    word_thres = 10
    vocab = LanguageIndex(documents, word_thres)
    pretrained_embedding = utils.get_pretained_glove(
        vocab.word2idx, word2embedding, embedding_dim
    )
    pretrained_embedding = pretrained_embedding.transpose((1,0))
    x_tr, m_tr, y_tr, x_te, m_te, y_te = load_train_test(documents, ky_ret, vocab)
    train_dataset = rnnDataset(x_tr, y_tr, m_tr)
    val_dataset = rnnDataset(x_te, y_te, m_te)

    # Load Market Model (need to define path to pretrained models)
    market_model = encoder.Encoder(pretrained_embedding, args)
    market_gen = generator.Generator(pretrained_embedding, args)
    market_model.load_state_dict(torch.load(args["market_model_path"]))
    market_gen.load_state_dict(torch.load(args['market_gen_path']))

    # Load Industry Model
    industry_model = encoder.Encoder(pretrained_embedding, args)
    industry_gen = generator.Generator(pretrained_embedding, args)
    industry_model.load_state_dict(torch.load(args["industry_model_path"]))
    industry_gen.load_state_dict(torch.load(args['industry_gen_path']))


    (
        industry_word_list, industry_rat_list, 
        industry_dir, mar_word_list, mar_rat_list, mar_dir
    ) = generate_rationale_list(
        val_dataset, market_model, market_gen, 
        industry_model, industry_gen, args
    )

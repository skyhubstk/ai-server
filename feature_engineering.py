def create_features(fin):

    revenue=fin['revenue_4']
    op=fin['op_4']
    debt=fin['debt_4']
    equity=fin['equity_4']

    fin['op_margin']=op/revenue
    fin['debt_ratio']=debt/equity

    return fin
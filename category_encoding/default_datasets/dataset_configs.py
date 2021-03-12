DEFAULT_DATASETS = {
    'telco': {
        'dataset': 'blastchar/telco-customer-churn',
        'file_name': 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'read_fargs': {'index_col': 0},
        'replace': {'TotalCharges': {' ': '0'}},
        'str_cols': [
            'gender',
            'SeniorCitizen',
            'Partner',
            'Dependents',
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod',
            'Churn'
        ],
        'cat_cols': [
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaymentMethod'
        ],
        'target_col': 'Churn'
    },
    'Adult income dataset': {
        'dataset': 'wenruliu/adult-income-dataset',
        'file_name': 'adult.csv'
    }
}
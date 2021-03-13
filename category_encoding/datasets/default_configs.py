DEFAULT_KAGGLE_CONFIGS = {
    'TelcoDataset': {
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
    'AdultDataset': {
        'str_cols': [
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'native-country',
            'income'
        ],
        'cat_cols': [
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'gender',
            'native-country'
        ],
        'target_col': 'income'
    },
    'EmployeeDataset': {
        'cat_cols': [
            'RESOURCE', 
            'MGR_ID', 
            'ROLE_ROLLUP_1', 
            'ROLE_ROLLUP_2',
            'ROLE_DEPTNAME',
            'ROLE_TITLE',
            'ROLE_FAMILY_DESC', 
            'ROLE_FAMILY',
            'ROLE_CODE'
        ],
        'target_col': 'ACTION'
    },
    'KickedDataset': {
        'str_cols': [
            'Auction',
            'Make',
            'Model',
            'Trim',
            'SubModel',
            'Color',
            'Transmission',
            'WheelType',
            'Nationality',
            'Size',
            'TopThreeAmericanName',
            'PRIMEUNIT',
            'AUCGUART',
            'VNST'
        ],
        'cat_cols': [
            'Auction',
            'Make',
            'Model',
            'Trim',
            'SubModel',
            'Color',
            'Transmission',
            'WheelType',
            'Nationality',
            'Size',
            'TopThreeAmericanName',
            'PRIMEUNIT',
            'AUCGUART',
            'VNST',
            'WheelTypeID',
            'VNZIP1'
        ],
        'target_col': 'IsBadBuy'
    },
    'CreditDataset': {
        'str_cols': [
            'NAME_CONTRACT_TYPE',
            'CODE_GENDER',
            'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY',
            'NAME_TYPE_SUITE',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE',
            'OCCUPATION_TYPE',
            'WEEKDAY_APPR_PROCESS_START',
            'ORGANIZATION_TYPE',
            'FONDKAPREMONT_MODE',
            'HOUSETYPE_MODE',
            'WALLSMATERIAL_MODE',
            'EMERGENCYSTATE_MODE'
        ],
        'cat_cols': [
            'CODE_GENDER',
            'REGION_RATING_CLIENT',
            'REGION_RATING_CLIENT_W_CITY',
            'NAME_TYPE_SUITE',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE',
            'OCCUPATION_TYPE',
            'WEEKDAY_APPR_PROCESS_START',
            'ORGANIZATION_TYPE',
            'FONDKAPREMONT_MODE',
            'HOUSETYPE_MODE',
            'WALLSMATERIAL_MODE',
            'EMERGENCYSTATE_MODE'
        ],
        'target_col': 'TARGET'
    }
}
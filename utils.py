def check_data(df):
    issues = []

    if df.isnull().sum().sum() > 0:
        issues.append("Dataset contains missing values")

    if df.duplicated().sum() > 0:
        issues.append("Dataset contains duplicate rows")

    if df.shape[0] < 50:
        issues.append("Dataset is too small")

    return issues
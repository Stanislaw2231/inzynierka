
def apply_rolling_mean(df, unique_sets, columns):
    for s in unique_sets:
        subset = df[df['set'] == s]
        for col in columns:
            df.loc[df['set'] == s, col + '_roll'] = subset[col].rolling(window=3, min_periods=1).mean()
    return df

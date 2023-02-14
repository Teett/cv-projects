#%%
import pysubgroup as ps

# Load the example dataset
from pysubgroup.tests.DataSets import get_titanic_data
data = get_titanic_data()

data.head()
# %% 
target = ps.BinaryTarget ('Survived', True)

searchspace = ps.create_selectors(data, ignore=['Survived'])

task = ps.SubgroupDiscoveryTask (
    data, 
    target, 
    searchspace, 
    result_set_size=5, 
    depth=2, 
    qf=ps.WRAccQF())

result = ps.BeamSearch().execute(task)

#%%
df_result = result.to_dataframe()
print(df_result)


# %%

import punctafinder

available_functions = [f for f in punctafinder.__dict__.keys()]

print("All available functions in punctafinder package :")
print('\n'.join(available_functions))

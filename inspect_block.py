import ultralytics.nn.modules.block as block
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"Block module path: {block.__file__}")
if hasattr(block, 'A2C2f'):
    print("A2C2f FOUND in block module.")
else:
    print("A2C2f NOT FOUND in block module.")
    print("Available attributes in block module:")
    print([x for x in dir(block) if not x.startswith('__')])

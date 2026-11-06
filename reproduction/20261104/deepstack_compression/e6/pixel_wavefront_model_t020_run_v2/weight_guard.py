"""Record payload file stat identity; SHA remains from the accepted preflight."""
from pathlib import Path

def snapshot(files):
 if len(files)!=4:raise ValueError('four weight shards required')
 out=[]
 for row in files:
  path=Path(row['path']);st=path.stat()
  if st.st_size!=row['bytes'] or row['sha256']!=row['expected_lfs_sha256']:raise ValueError('weight metadata mismatch')
  out.append({'path':str(path),'dev':st.st_dev,'ino':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'expected_sha256':row['expected_lfs_sha256']})
 return out

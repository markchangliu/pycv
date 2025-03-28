import os
from typing import Union, List, Generator


def load_files(
    search_dirs: Union[str, os.PathLike],
    include_prefixes: List[str],
    include_suffixes: List[str],
    include_tags: List[str],
    exclude_prefixes: List[str],
    exclude_suffixes: List[str],
    exclude_tags: List[str]
) -> Generator[Union[str, os.PathLike], None, None]:
    for search_dir in search_dirs:
        for root, subdirs, filenames in os.walk(search_dir):
            for filename in filenames:
                filename: str = filename
                file_p = os.path.join(root, filename)

                exclude_flag = False
                
                if filename.endswith(exclude_suffixes):
                    exclude_flag = True
                if filename.startswith(exclude_prefixes):
                    exclude_flag = True
                for exclude_tag in exclude_tags:
                    if exclude_tag in filename:
                        exclude_flag = True
                        break
                
                if exclude_flag:
                    continue

                if len(include_suffixes) == 0 and \
                    len(include_prefixes) == 0 and \
                    len(include_tags) == 0:
                    yield file_p
                
                include_flag = False

                if len(include_prefixes) > 0:
                    if filename.startswith(include_prefixes):
                        include_flag = True
                if len(include_suffixes) > 0:
                    if filename.endswith(include_suffixes):
                        include_flag = True
                if len(include_tags) > 0:
                    for include_tag in include_tags:
                        if include_tag in filename:
                            include_flag = True
                            break
                
                if include_flag:
                    yield file_p


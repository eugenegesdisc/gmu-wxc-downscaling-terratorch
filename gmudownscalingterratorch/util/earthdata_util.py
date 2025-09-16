import yaml

def read_earthdata_authentication(
        earthdata_login_config:str)->tuple[str|None,str|None,str|None]:
    username = None
    password = None
    edl_token = None
    with open(earthdata_login_config, 'r') as file:
        data = yaml.safe_load(file)
        if "edl_token" in data:
            edl_token = data["edl_token"]
        if "username" in data:
            username = data["username"]
        if "password" in data:
            password = data["password"]
    return username, password, edl_token

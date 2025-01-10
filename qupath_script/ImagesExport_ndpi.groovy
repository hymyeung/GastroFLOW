
//if (getCurrentServer().getPath().contains("--series, 0")){
def out_path = getCurrentServer().getPath()
slash_pointer = out_path.lastIndexOf('/');
dot_pointer = out_path.lastIndexOf('.');
int length  = dot_pointer - slash_pointer
out_path = out_path.substring(slash_pointer+1, slash_pointer+length);
print(out_path)
path="/home/ray/Desktop/GastroFLOW/GastroFLOW/Feature_dataset/20231010-PolyU-PYNEH_slide/snapshot/" // Please Change the Path
def oriout=path+out_path+".png"

def server = getCurrentServer()
def requestFull = RegionRequest.createInstance(server,8)
writeImageRegion(server, requestFull, oriout)

print("Done!")
//}
//else{
//print("Skipping macro .scn file: "+getCurrentServer().getPath())
//}
print("finished")
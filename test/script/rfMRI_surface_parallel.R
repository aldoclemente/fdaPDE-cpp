library(fdaPDE)
subjectsID = as.matrix(read.csv("data/rfMRI_surface/subjectsIDs.csv", header = T)[,2])
points = as.matrix(read.csv("../data/mesh/brain_lh_surface_32k/points.csv")[,2:4])
elements = as.matrix(read.csv("../data/mesh/brain_lh_surface_32k/elements.csv")[,2:4])
mesh = create.mesh.2.5D(nodes=points, triangles = elements) 
fembasis = create.FEM.basis(mesh)
f = matrix(0,nrow = fembasis$nbasis , ncol=nrow(subjectsID))
beta = matrix(0, nrow=1, ncol=nrow(subjectsID))

outdir = "data/rfMRI_surface/results/"
f_file = paste0(outdir, subjectsID, ".f.txt")
beta_file = paste0(outdir, subjectsID, ".beta.txt")

for(i in 1:nrow(subjectsID)){
  f[,i] = as.matrix(read.table(f_file[i] ,header = F))
  beta[,i] = as.matrix(read.table(beta_file[i], header = F))
}

f_mean = apply(f, MARGIN = 1, FUN=mean)
beta_mean = apply(beta, MARGIN =1, FUN=mean)

imgsdir = "data/rfMRI_surface/imgs/"
if(!dir.exists(imgsdir)) dir.create(imgsdir)

# !!! path/to/graphic-tools ----------------------------------------------------
{
path_ = "~/Desktop/graphic-tools/"
source(paste0(path_, "utils.R"))
ll <- parse(file = paste0(path_, "plot_smooth_2.5D.R"))

for (i in seq_along(ll)) {
  tryCatch(eval(ll[[i]]), 
           error = function(e) message(as.character(e)))
}
}
# ------------------------------------------------------------------------------
#na_mask = as.matrix(read.csv("data/rfMRI_surface/na_mask.csv")[,2])
na_mask = is.na(as.matrix(read.csv("data/rfMRI_surface/FCmaps/100307.fc_map.csv")[,2]))
f_mean[na_mask] = NA

plot_smooth_2.5D(FEM(f_mean, fembasis), colorscale = viridis)
snapshot3d(filename = paste0(imgsdir,"f_mean.png"),
           fmt = "png", width = 800, height = 750, webshot = rgl.useNULL())
close3d()  

plot_colorbar(FEM(f_mean, fembasis), colorscale = viridis,
              file = paste0(imgsdir, "colorbar_f_mean"))

plot_mesh_2.5D <- function(mesh, ROI=NULL, NA_ = NULL,...){
  
  dummyFEM = FEM(rep(0, nrow(mesh$nodes)), create.FEM.basis(mesh))
  dummyFEM$coeff[ROI,] <- 1 
  dummyFEM$coeff[NA_,] <- 2
  
  
  #if (is.null(m)) { 
  m = min(dummyFEM$coeff)
  #}
  #if (is.null(M)) { 
  M = max(dummyFEM$coeff)
  #}
  triangles = c(t(dummyFEM$FEMbasis$mesh$triangles))
  ntriangles = nrow(dummyFEM$FEMbasis$mesh$triangles)
  order = dummyFEM$FEMbasis$mesh$order
  nodes = dummyFEM$FEMbasis$mesh$nodes
  edges = matrix(rep(0, 6*ntriangles), ncol = 2)
  for(i in 0:(ntriangles-1)){
    edges[3*i+1,] = c(triangles[3*order*i+1], triangles[3*order*i+2])
    edges[3*i+2,] = c(triangles[3*order*i+1], triangles[3*order*i+3])
    edges[3*i+3,] = c(triangles[3*order*i+2], triangles[3*order*i+3])
  }
  edges = edges[!duplicated(edges),]
  edges <- as.vector(t(edges))
  
  coeff = dummyFEM$coeff
  
  FEMbasis = dummyFEM$FEMbasis
  
  #p = jet.col(n = 1000, alpha = 0.8)
  # alternative color palette: p <- colorRampPalette(c("#0E1E44", "#3E6DD8", "#68D061", "#ECAF53", "#EB5F5F", "#E11F1C"))(1000)
  if(!is.null(ROI) & !is.null(NA_)){
    p = c("lightgray", "red3", "blue3")
  }else{ 
    p = c("lightgray", "blue3")
  }
  palette(p)
  
  ncolor = length(p)
  
    open3d(zoom = zoom, userMatrix = userMatrix, windowRect = windowRect)
    rgl.pop("lights") 
    light3d(specular = "black") 
    
    diffrange = M - m
    
    col = coeff[triangles]
    col = (col - min(coeff, na.rm = T))/diffrange*(ncolor-1)+1
    
    rgl.triangles(x = nodes[triangles ,1], y = nodes[triangles ,2],
                  z = nodes[triangles,3],
                  color = col,...)
    # rgl.lines(x = nodes[edges ,1], y = nodes[edges ,2],
    #           z = nodes[edges,3],
    #           color = "black",...)
    aspect3d("iso")
}

plot_mesh_2.5D(mesh, NA_ = na_mask)
snapshot3d(filename = paste0(imgsdir,"mesh.png"),
           fmt = "png", width = 800, height = 750, webshot = rgl.useNULL())
close3d()  

### --------------- nonparam test ----------------------------------------------



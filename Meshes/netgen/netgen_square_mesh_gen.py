from netgen.geom2d import unit_square, MakeCircle, SplineGeometry
from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, FaceDescriptor, Mesh
from netgen.csg import Pnt

def GenSquareMesh(N):
    mesh = Mesh()
    # mesh.SetGeometry(unit_square)
    geom = SplineGeometry("square.in2d")
    mesh.dim = 2

    pnums = []
    for i in range(N + 1):
        for j in range(N + 1):
            pnums.append(mesh.Add(MeshPoint(Pnt(i / N, j / N, 0))))

    mesh.SetMaterial(1, "mat")
    mesh.Add (FaceDescriptor(surfnr=1,domin=0,bc=1))

    for j in range(N):
        for i in range(N):
            mesh.Add(Element2D(1, [pnums[i + j * (N + 1)], pnums[i + (j + 1) * (N + 1)], pnums[i + 1 + j * (N + 1)]]))
            mesh.Add(Element2D(1, [pnums[i + (j + 1) * (N + 1)], pnums[i + 1 + (j + 1) * (N + 1)], pnums[i + 1 + j * (N + 1)]]))


    for i in range(N):
       mesh.Add(Element1D([pnums[N + i * (N + 1)], pnums[N + (i + 1) * (N + 1)]], index=1))
       mesh.Add(Element1D([pnums[0 + i * (N + 1)], pnums[0 + (i + 1) * (N + 1)]], index=1))


    for i in range(N):
       mesh.Add(Element1D([pnums[i], pnums[i + 1]], index=1))
       mesh.Add(Element1D([pnums[i + N * (N + 1)], pnums[i + 1 + N * (N + 1)]], index=1))

    f_name = "square"+str(N*N*2)+".vol"
    mesh.Save(f_name)

for N in range(3, 33):
    print(N)
    GenSquareMesh(N)
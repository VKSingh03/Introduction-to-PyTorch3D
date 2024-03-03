"""
Assignment 1 for CMSC848F

Submission By: 
Name: Vineet Singh 
UID: 119123614
"""

import argparse
import pickle
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
import mcubes
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh, get_points_renderer, unproject_depth_image

# import for rendering gif
import imageio

def render_gif_from_mesh(output_path, verts, faces, textures, distance = 4.0,image_size=256,device = None, gif = True):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)
    
    mesh = pytorch3d.structures.Meshes(
        verts=verts,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    if gif == True:
        images = []
        for i in tqdm(range(0, 360,10)):

            R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist =distance, 
                                                                    azim = i, 
                                                                    device=device)
            
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R, T=T, fov=60, device=device
            )

            rend = renderer(mesh, cameras=cameras, lights=lights)
            rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
            # The .cpu moves the tensor to GPU (if needed).
            rend =  (rend * 255).astype(np.uint8)
            images.append(rend)
        imageio.mimsave(output_path, images, duration=65, loop=0)
    else: 
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
        )

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        plt.imsave(output_path, rend)

def render_cow(cow_path="data/cow.obj", output_path = "results/cow_render.gif",color=[0.7, 0.7, 1],image_size=256, output = "gif"):
    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    if output == "gif":
        render_gif_from_mesh(output_path=output_path,verts=vertices, faces=faces, textures=textures,image_size=image_size)
    else:
        render_gif_from_mesh(output_path=output_path,verts=vertices, faces=faces, textures=textures,image_size=image_size,gif=False)

def dolly_zoom(
    image_size=256,
    num_frames=10,
    duration=3,
    device=None,
    output_file="output/dolly.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)

    renders = []
    for fov in tqdm(fovs):
        distance = 2.2 / (torch.tan(torch.deg2rad(fov / 2)))  # TODO: change this.
        T = [[0, 0, distance]]  # TODO: Change this.
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=duration, loop = 0)

def render_shape(shape, image_size=256, output_path = "results/tetrahederon_render.gif",color=[0.7, 0.7, 1]):

    if(shape == 'tetra'):
        vertices = torch.tensor([[1,-1,0],
                                [-1,-1,0],
                                [0,-1,1.732],
                                [0,0.22,0.512]])
        vertices = vertices.unsqueeze(0)
        
        faces = torch.tensor([[0,1,2],
                            [0,2,3],
                            [1,2,3],
                            [0,1,3]])
        faces = faces.unsqueeze(0)

    else:
        vertices = torch.tensor([[1,-1,-1],
                                [-1,-1,-1],
                                [-1,1,-1],
                                [1,1,-1],
                                [1,1,1],
                                [-1,1,1],
                                [-1,-1,1],
                                [1,-1,1]], dtype=torch.float32)
        vertices = vertices.unsqueeze(0)
        
        faces = torch.tensor([[0,1,2],
                              [0,2,3],
                              [0,3,4],
                              [0,4,7],
                              [4,5,6],
                              [4,6,7],
                              [1,2,5],
                              [1,5,6],
                              [0,1,6],
                              [0,6,7],
                              [2,3,4],
                              [2,4,5]], dtype=torch.int64)
        faces = faces.unsqueeze(0)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)
    render_gif_from_mesh(output_path=output_path,verts=vertices,faces=faces,textures=textures, image_size=image_size)

def render_cow_retexture(
    cow_path="data/cow.obj", image_size=256, output_path = "results/cow_render_textured.gif",color=[0.7, 0.7, 1]):
    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)

    # Extracting the z-coordinates of the mesh. 
    z = vertices[:,2]
    color1 = torch.tensor([1, 0.5, 0])
    color2 = torch.tensor([0.1, 1, 0])
    alpha = (z - z.min()) / (z.max() - z.min())
    alpha = alpha.repeat(3,1).T
    color = alpha * color2 + (1 - alpha) * color1

    textures = textures * (color)
    textures = textures.unsqueeze(0) # (1, N_v, 3)
    
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    render_gif_from_mesh(output_path=output_path,verts=vertices,faces=faces,textures=textures)
 
def render_cow_transformations(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[0, 0, -1], [0, 1, 0], [1.0, 0, 0]],
    T_relative=[3, 0, 3],
    device=None):

    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()

def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_gif_from_pc(points, colors, output_path, image_size=256,background_color=(1, 1, 1),device=None, distance = 10):
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    point_cloud = pytorch3d.structures.Pointclouds(points=points,
                                                   features=colors).to(device)

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)
    
    images = []
    for i in tqdm(range(0, 360, 10)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=distance,
                                                                 azim=i,
                                                                 device=device)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        rend = renderer(point_cloud, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        rend = (rend * 255).astype(np.uint8)
        images.append(rend)

    imageio.mimsave(output_path, images, duration = 60, loop = 0)

def render_plant(
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    
    data = load_rgbd_data(path="data/rgbd_data.pkl")
    points1, colors1 = unproject_depth_image(torch.from_numpy(data['rgb1']),
                                       torch.from_numpy(data['mask1']),
                                       torch.from_numpy(data['depth1']),
                                       data['cameras1'])
    
    points2, colors2 = unproject_depth_image(torch.from_numpy(data['rgb2']),
                                       torch.from_numpy(data['mask2']),
                                       torch.from_numpy(data['depth2']),
                                       data['cameras2'])
    # The points are rotated, so rotating them back to correct orientation.
    R = torch.tensor([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]], dtype=torch.float32)
    points1 = torch.matmul(points1, R)
    points2 = torch.matmul(points2, R)
    points3 = torch.cat((points1, points2), dim=0)
    colors3 = torch.cat((colors1, colors2), dim=0)
    points1.unsqueeze_(0)
    colors1.unsqueeze_(0)
    points2.unsqueeze_(0)
    colors2.unsqueeze_(0)
    print("Rendering first point cloud")
    render_gif_from_pc(points1, colors1, output_path='results/plant1.gif', device=device)
    print("Rendering second point cloud")
    render_gif_from_pc(points2, colors2, output_path='results/plant2.gif', device=device)
    
    points3.unsqueeze_(0)
    colors3.unsqueeze_(0)    
    print("Rendering Merged point cloud")
    render_gif_from_pc(points3, colors3, output_path='results/plant3.gif', device=device)

def render_torus(image_size=256, num_samples=1000, device=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """
    if device is None:
        device = get_device()

    R = 3 # outer radius
    r = 1 # inner radius

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R + r*torch.cos(Theta))*torch.cos(Phi)
    y = (R + r*torch.cos(Theta))*torch.sin(Phi)
    z = r*torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],device=device)
    
    images = []
    for i in tqdm(range(0, 360, 10)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=10,
                                                                 azim=i,
                                                                 device=device)

        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(point_cloud, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy()
        rend = (rend * 255).astype(np.uint8)
        images.append(rend)
    imageio.mimsave("results/torus.gif", images, duration = 65, loop = 0)

def render_torus_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()

    R = 2 # outer radius
    r = 1 # inner radius

    min_value = -3.1
    max_value = 3.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (torch.sqrt(X**2 +Y**2) - R)**2 + Z**2 -r**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # 
    images = []
    for i in tqdm(range(0, 360, 10)):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=10,
                                                                 azim=i,
                                                                 device=device)
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy()
        rend = (rend * 255).astype(np.uint8)
        images.append(rend)
    imageio.mimsave("results/torus_implicit.gif", images, duration = 65, loop = 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="1.1", 
                        choices=['0.1','1.1','1.2','2.1','2.2','3','4','5.1','5.2','5.3'])
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    if(args.question == "0.1"):
        render_cow(image_size=args.image_size, output_path = "results/cow_render.jpg", output = "image")
    elif(args.question == "1.1"):
        render_cow(image_size=args.image_size, output_path = "results/cow_render.gif")
    elif(args.question == "1.2"):
        dolly_zoom(image_size=args.image_size, output_file="results/dolly.gif", duration=3)
    elif(args.question == "2.1"):
        render_shape(shape="tetra", image_size=args.image_size, output_path="results/tetrahederon_render.gif")
    elif(args.question == "2.2"):
        render_shape(shape="cube", image_size=args.image_size, output_path="results/cube_render.gif")
    elif(args.question == "3"):
        render_cow_retexture(image_size=args.image_size, output_path="results/cow_render_textured.gif")
    elif(args.question == "4"):
        plt.imsave("results/transform_cow1.jpg", render_cow_transformations(image_size=args.image_size, 
                                                                            R_relative=[[0, -1, 0], 
                                                                                        [1, 0, 0], 
                                                                                        [0, 0, 1]],  
                                                                            T_relative=[0, 0, 0]))
        plt.imsave("results/transform_cow2.jpg", render_cow_transformations(image_size=args.image_size,  
                                                                            R_relative=[[1, 0, 0], 
                                                                                        [0, 1, 0], 
                                                                                        [0, 0, 1]], 
                                                                            T_relative=[0, 0, 2]))
        plt.imsave("results/transform_cow3.jpg", render_cow_transformations(image_size=args.image_size,  
                                                                            R_relative=[[1, 0, 0], 
                                                                                        [0, 1, 0], 
                                                                                        [0, 0, 1]], 
                                                                            T_relative=[0.5, -0.5, 0]))
        plt.imsave("results/transform_cow4.jpg", render_cow_transformations(image_size=args.image_size, 
                                                                            R_relative=[[0, 0, -1], 
                                                                                        [0, 1, 0], 
                                                                                        [1, 0, 0]], 
                                                                            T_relative=[3, 0, 3]))
        plt.imsave("results/transform_cow5.jpg", render_cow_transformations(image_size=args.image_size, 
                                                                            R_relative=[[0, -1, 0], 
                                                                                        [1, 0, 0], 
                                                                                        [0, 0, 1]], 
                                                                            T_relative=[0.5, -0.5, 0]))
    elif(args.question == "5.1"):
        render_plant(image_size=args.image_size)
    elif(args.question == "5.2"):
        render_torus(image_size=args.image_size, num_samples=200)
    elif(args.question == "5.3"):
        image = render_torus_mesh(image_size=args.image_size)
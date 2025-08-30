type vec = (float * float * float)

type color = {
  r: int;
  g: int;
  b: int;
}

type ray_hit = {
  t: float;
  normal: vec;
}

type image = {
  width: int;
  height: int;
  data: color array array
}

type obj = {
  vertices: vec array;
  faces: (int*int*int) list;
  colors: color array
}

let cross (ax, ay, az) (bx, by, bz) : vec =
  let cx = ay *. bz -. az *. by in
  let cy = az *. bx -. ax *. bz in
  let cz = ax *. by -. ay *. bx in
  (cx, cy, cz)

let dot (ax, ay, az) (bx, by, bz) : float =
  ax *. bx +. ay *. by +. az *. bz
  
let sub (ax, ay, az) (bx, by, bz) : vec =
  (ax -. bx, ay -. by, az -. bz)

let add (ax, ay, az) (bx, by, bz) : vec =
  (ax +. bx, ay +. by, az +. bz)

let scale (s: float) (ax, ay, az) : vec =
  (ax *. s, ay *. s, az *. s)

let length (ax, ay, az) : float =
  sqrt (ax *. ax +. ay *. ay +. az *. az)

let normalize (ax, ay, az) : vec =
  let len = length (ax, ay, az) in
  (ax /. len, ay /. len, az /. len)

let pi = 3.14159265358979323846

let ray_triangle_intersection (origin: vec) (direction: vec) (triangle: vec * vec * vec) : ray_hit option =
  let (a, b, c) = triangle in
  let e1 = sub b a in
  let e2 = sub c a in
  let p = cross direction e2 in
  let det = dot e1 p in
  if det < 1e-6 then None else
  let inv_det = 1.0 /. det in
  let t = sub origin a in
  let u = dot t p *. inv_det in
  if u < 0.0 || u > 1.0 then None else
  let q = cross t e1 in
  let v = dot direction q *. inv_det in
  if v < 0.0 || u +. v > 1.0 then None else
  let t = dot e2 q *. inv_det in
  let n = normalize (cross e1 e2) in
  Some {t; normal=n}

let read_obj (filename: string) (color:color) : obj =
  let face_index str =
    let str = String.trim str in
    let tokens = String.split_on_char '/' str in
    let idx = int_of_string (List.nth tokens 0) in
    idx - 1
  in
  let vertex str = 
    let str = String.trim str in
    try
      float_of_string str
    with
      _ -> failwith ("Invalid vertex: " ^ str)
  in
  let ic = open_in filename in
  let vertices = ref [] in
  let faces = ref [] in
  (try
    while true do
      let line = input_line ic in
      if line = "" then () else
      let tokens = String.split_on_char ' ' line in
      match tokens with
      | "v" :: x :: y :: z :: _ -> vertices := (vertex x, vertex y, vertex z) :: !vertices
      | "f" :: f1 :: f2 :: f3 :: _ -> faces := (face_index f1, face_index f2, face_index f3) :: !faces
      | _ -> ()
    done;
    close_in ic
  with
    End_of_file ->
      close_in ic);
  {
    vertices = Array.of_list (List.rev !vertices);
    faces = List.rev !faces;
    colors = Array.make (List.length !faces) color
  }


let calculate_camera_basis (direction: vec) : vec * vec * vec =
  let forward = normalize direction in
  let world_up = (0.0, 1.0, 0.0) in
  let right = cross world_up forward in
  let len = length right in
  let world_up' = if len < 1e-6 then (1.0, 0.0, 0.0) else world_up in
  let right' = cross world_up' forward in
  let right' = normalize right' in
  let up = cross forward right' in
  let up = normalize up in
  let up = scale (-1.0) up in
  let right' = scale (-1.0) right' in
  (forward, right', up)

let render (width: int) (height: int) (objects: obj list) (origin: vec) (direction: vec) (fov: float) : image * float array array =

  let light_dir = normalize (0.0, 0.0, 1.0) in

  let depth_map = Array.make_matrix width height 1000000.0 in
  let image = {
    width;
    height;
    data = Array.make_matrix width height {r=0; g=0; b=0}
  } in

  let forward, right, up = calculate_camera_basis direction in

  let aspect_ratio = float_of_int width /. float_of_int height in
  let fov_rad = fov *. pi /. 180.0 in
  let half_tan_fov = tan (fov_rad *. 0.5) in

  for i = 0 to width - 1 do
    for j = 0 to height - 1 do
      let x = (2.0 *. (float_of_int i +. 0.5) /. float_of_int width -. 1.0) *. aspect_ratio *. half_tan_fov in
      let y = (2.0 *. (float_of_int j +. 0.5) /. float_of_int height -. 1.0) *. half_tan_fov in
      let ray_direction = add forward (add (scale x right) (scale y up)) in
      let ray_direction = normalize ray_direction in
      let closest_t = ref Float.infinity in
      let closest_color = ref {r=0; g=0; b=0} in
      List.iter (fun {vertices; faces; colors} ->
        List.iteri (fun face_idx face ->
          let idx0, idx1, idx2 = face in
          let a = vertices.(idx0) in
          let b = vertices.(idx1) in
          let c = vertices.(idx2) in
          let triangle = (a, b, c) in
          let t = ray_triangle_intersection origin ray_direction triangle in
          match t with
          | Some {t; normal=n} when t > 0.0 && t < !closest_t ->
            closest_t := t;
            (* closest_color := colors.(face_idx) *)
            (* phong-like diffuse shading *)
            let d = max 0.4 (dot n light_dir) in
            let {r; g; b} = colors.(face_idx) in
            closest_color := {
              r=int_of_float (float_of_int r *. d);
              g=int_of_float (float_of_int g *. d);
              b=int_of_float (float_of_int b *. d)
            }
          | _ -> ()
        ) faces
      ) objects;
      if !closest_t < Float.infinity then begin
        image.data.(i).(j) <- !closest_color;
        depth_map.(i).(j) <- !closest_t
      end
    done
  done;
  (image, depth_map)

let save_image (image: image) (filename: string) : unit =
  let ic = open_out filename in
  output_string ic "P3\n";
  output_string ic (string_of_int image.width ^ " " ^ string_of_int image.height ^ "\n");
  output_string ic "255\n";
  for j = 0 to image.height - 1 do
    for i = 0 to image.width - 1 do
      let {r; g; b} = image.data.(i).(j) in
      output_string ic (string_of_int r ^ " " ^ string_of_int g ^ " " ^ string_of_int b ^ "\n")
    done
  done;
  close_out ic


let create_debug_cube (size: float) : obj =
  let half_size = size /. 2.0 in
  let vertices = [|
    (-.half_size, -.half_size, -.half_size);
    (  half_size, -.half_size, -.half_size);
    (  half_size,   half_size, -.half_size);
    (-.half_size,   half_size, -.half_size);
    (-.half_size, -.half_size,   half_size);
    (  half_size, -.half_size,   half_size);
    (  half_size,   half_size,   half_size);
    (-.half_size,   half_size,   half_size);
  |] in
  let faces = [
    (0,2,1); (0,3,2);
    (4,5,6); (4,6,7);
    (0,4,7); (0,7,3);
    (1,2,6); (1,6,5);
    (0,1,5); (0,5,4);
    (2,3,7); (2,7,6);
  ] in
  let colors = Array.map (fun (r, g, b) -> {r; g; b}) [|
    (0,0,255); (0,0,255);
    (0,255,0); (0,255,0);
    (255,0,0); (255,0,0);
    (255,255,0); (255,255,0);
    (255,0,255); (255,0,255);
    (0,255,255); (0,255,255);
  |] in
  {
    vertices = vertices;
    faces = faces;
    colors = colors
  }


let width = 100
let height = 100

let origin = (2.0, 10.0, 8.0)
let destination = (0.0, 5.0, 0.0)
let fov = 90.0

(* let objects = [create_debug_cube 8.0] *)
let bunny = read_obj "data/bunny.obj" {r=255; g=0; b=0}
let bunny = {bunny with vertices = Array.map (scale 50.0) bunny.vertices}
let objects = [bunny]

let direction = normalize (sub destination origin)

let image, depth_map = render width height objects origin direction fov

let () = save_image image "output/ocaml.ppm"
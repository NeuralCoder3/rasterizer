{-# LANGUAGE RecordWildCards #-}

import System.IO (readFile)
import Data.List (isPrefixOf)
import Data.Char (isSpace)
import Data.List (minimumBy)
import Data.Maybe (mapMaybe)
import Data.Ord (comparing)
import Data.Array.IArray
import System.IO.Unsafe (unsafePerformIO)
type Scalar = Float

class Vector v where
  fromScalar :: Scalar -> v
  vmap       :: (Scalar -> Scalar) -> v -> v
  vzip       :: (Scalar -> Scalar -> Scalar) -> v -> v -> v
  vfold      :: (Scalar -> Scalar -> Scalar) -> v -> Scalar

vdot :: Vector v => v -> v -> Scalar
vdot v w = vfold (+) $ vzip (*) v w

vlen :: Vector v => v -> Scalar
vlen v = sqrt $ v `vdot` v

(|*) :: Vector v => Scalar -> v -> v
s |* v = vmap (*s) v
(*|) :: Vector v => v -> Scalar -> v
v *| s = vmap (*s) v

vnormalise :: Vector v => v -> v
vnormalise v =
  let m = vlen v
  in if m < 1e-10 then v else v *| (1/m)

data Vector3 = Vector3 {v3x, v3y, v3z :: {-# UNPACK #-} !Scalar}
  deriving (Eq, Ord, Show)

instance Vector Vector3 where
  fromScalar x = Vector3 x x x
  vmap  f (Vector3 x1 y1 z1)                    = Vector3 (f x1)    (f y1)    (f z1)
  vzip  f (Vector3 x1 y1 z1) (Vector3 x2 y2 z2) = Vector3 (f x1 x2) (f y1 y2) (f z1 z2)
  vfold f (Vector3 x1 y1 z1)                    = f x1 (f y1 z1)

instance Num Vector3 where
  (+) = vzip (+)
  (-) = vzip (-)
  (*) = vzip (*)
  negate = vmap negate
  abs    = vmap abs
  signum = vmap signum
  fromInteger n = let s = fromInteger n in fromScalar s

vcross :: Vector3 -> Vector3 -> Vector3
vcross (Vector3 x1 y1 z1) (Vector3 x2 y2 z2) = Vector3
  {
    v3x = y1 * z2   -   y2 * z1,
    v3y = z1 * x2   -   z2 * x1,
    v3z = x1 * y2   -   x2 * y1
  }

data Color = Color {
  r :: Int,
  g :: Int,
  b :: Int
}
  deriving (Eq, Ord, Show)

data RayHit = RayHit {
  t :: Scalar,
  normal :: Vector3
}

data Obj = Obj {
  vertices :: Array Int Vector3,
  faces :: [(Int, Int, Int)],
  colors :: Array Int Color
}

ray_triangle_intersection :: Vector3 -> Vector3 -> (Vector3, Vector3, Vector3) -> Maybe RayHit
ray_triangle_intersection origin direction (a,b,c) = 
  let e1 = b - a in
  let e2 = c - a in
  let p = direction `vcross` e2 in
  let det = e1 `vdot` p in
  if det < 1e-6 then Nothing else
  let inv_det = 1/det in
  let t = origin - a in
  let u = (t `vdot` p) * inv_det in
  if u < 0 || u > 1 then Nothing else
  let q = t `vcross` e1 in
  let v = (direction `vdot` q) * inv_det in
  if v < 0 || u + v > 1 then Nothing else
  let t2 = (e2 `vdot` q) * inv_det in
  Just (RayHit {t = t2, normal = vnormalise (e1 `vcross` e2)})

calculate_camera_basis :: Vector3 -> (Vector3, Vector3, Vector3)
calculate_camera_basis direction =
  let forward = vnormalise direction in
  let world_up = Vector3 0.0 1.0 0.0 in
  let right = vcross world_up forward in
  let len = vlen right in
  let world_up2 = if len < 1e-6 then Vector3 1.0 0.0 0.0 else world_up in
  let right2 = vcross world_up2 forward in
  let right3 = vnormalise right2 in
  let up = vcross forward right3 in
  let up2 = vnormalise up in
  let up3 = up2 *| (-1.0) in
  let right4 = right3 *| (-1.0) in
  (forward, right4, up3)

data Image = Image {
  width :: Int,
  height :: Int,
  pixels :: [[Color]]
}


render :: Int -> Int -> [Obj] -> Vector3 -> Vector3 -> Scalar -> (Image, [[Scalar]])
render width height objects origin direction fov =
  let
    lightDir = vnormalise (Vector3 0 0 1)
    (forward, right, up) = calculate_camera_basis direction

    aspectRatio = fromIntegral width / fromIntegral height
    fovRad = fov * pi / 180.0
    halfTanFov = tan (fovRad * 0.5)

    -- compute one pixel
    pixel i j =
      let
        x = (2 * ((fromIntegral i + 0.5) / fromIntegral width) - 1)
              * aspectRatio * halfTanFov
        y = (2 * ((fromIntegral j + 0.5) / fromIntegral height) - 1)
              * halfTanFov
        rayDir = vnormalise $ forward + (x |* right) + (y |* up)

        hits = concatMap (intersections rayDir) objects
        closest = if null hits then Nothing else Just (minimumBy (comparing fst) hits)

      in case closest of
           Just (t, c) -> (c, t)
           Nothing     -> (Color 0 0 0, 1/0)  -- infinity
      where
        intersections rayDir Obj{..} =
          [ (t, shade normal (colors ! faceIdx))
          | (faceIdx, (i0,i1,i2)) <- zip [0..(length faces - 1)] faces
          , let a = vertices ! i0
                b = vertices ! i1
                c = vertices ! i2
                tri = (a,b,c)
          , Just RayHit{..} <- [ray_triangle_intersection origin rayDir tri]
          , t > 0
          ]
        shade n col@Color{..} =
          let d = max 0.4 (vdot n lightDir)
          in Color
               { r = floor (fromIntegral r * d)
               , g = floor (fromIntegral g * d)
               , b = floor (fromIntegral b * d)
               }
    
    -- build full grid
    pixels = [ [ pixel i j | i <- [0 .. width-1] ] | j <- [0 .. height-1] ]
    (colors, depths) = unzip (map unzip pixels)

  in (Image width height colors, depths)


mkArray :: [a] -> Array Int a
mkArray xs = array (0, length xs - 1) (zip [0..] xs)

create_debug_cube :: Scalar -> Obj
create_debug_cube size =
  let halfSize = size / 2

      vertices =
        [ Vector3 (-halfSize) (-halfSize) (-halfSize)
        , Vector3   halfSize  (-halfSize) (-halfSize)
        , Vector3   halfSize    halfSize  (-halfSize)
        , Vector3 (-halfSize)   halfSize  (-halfSize)
        , Vector3 (-halfSize) (-halfSize)   halfSize
        , Vector3   halfSize  (-halfSize)   halfSize
        , Vector3   halfSize    halfSize    halfSize
        , Vector3 (-halfSize)   halfSize    halfSize
        ]

      faces =
        [ (0,2,1), (0,3,2)
        , (4,5,6), (4,6,7)
        , (0,4,7), (0,7,3)
        , (1,2,6), (1,6,5)
        , (0,1,5), (0,5,4)
        , (2,3,7), (2,7,6)
        ]

      colors =
        [ Color   0   0 255, Color   0   0 255
        , Color   0 255   0, Color   0 255   0
        , Color 255   0   0, Color 255   0   0
        , Color 255 255   0, Color 255 255   0
        , Color 255   0 255, Color 255   0 255
        , Color   0 255 255, Color   0 255 255
        ]

  in Obj { vertices = mkArray vertices, faces = faces, colors = mkArray colors }


save_image :: Image -> FilePath -> IO ()
save_image img filename = do
  let w = width img
      h = height img
      pxs = pixels img
      header = "P3\n" ++ show w ++ " " ++ show h ++ "\n255\n"
      body = unlines
        [ unwords [show (r c) ++ " " ++ show (g c) ++ " " ++ show (b c) | c <- row]
        | row <- pxs
        ]
  writeFile filename (header ++ body)

-- Trim whitespace (like OCaml's String.trim)
trim :: String -> String
trim = f . f
  where f = reverse . dropWhile isSpace

-- Parse "42/..." -> 41 (OBJ indices are 1-based)
faceIndex :: String -> Int
faceIndex str =
  case wordsWhen (== '/') (trim str) of
    (tok:_) -> read tok - 1
    []      -> error $ "Invalid face token: " ++ str

-- Parse a vertex coordinate (with error handling)
vertexCoord :: String -> Scalar
vertexCoord s =
  case reads (trim s) of
    [(x,"")] -> x
    _        -> error ("Invalid vertex: " ++ s)

-- Split by a delimiter (like OCaml's String.split_on_char)
wordsWhen :: (Char -> Bool) -> String -> [String]
wordsWhen p s =
  case dropWhile p s of
    "" -> []
    s' -> w : wordsWhen p s''
          where (w, s'') = break p s'

read_obj :: FilePath -> Color -> IO Obj
read_obj filename color = do
  contents <- readFile filename
  let ls = lines contents

      parseLine (vs, fs) line =
        case words line of
          ("v":x:y:z:_) ->
            (Vector3 (vertexCoord x) (vertexCoord y) (vertexCoord z) : vs, fs)
          ("f":a:b:c:_) ->
            (vs, (faceIndex a, faceIndex b, faceIndex c) : fs)
          _ -> (vs, fs)

      (vs, fs) = foldl parseLine ([], []) ls
      vs' = reverse vs
      fs' = reverse fs
      cols = replicate (length fs') color

  pure $ Obj { vertices = mkArray vs', faces = fs', colors = mkArray cols }



main:: IO ()
main = do
  let width = 100
  let height = 100

  let origin :: Vector3 = Vector3 2.0 10.0 8.0
  let destination = Vector3 0.0 5.0 0.0
  let direction = vnormalise (destination - origin)
  let fov = 90.0

  -- let objects = [create_debug_cube 8.0]
  bunny <- read_obj "data/bunny.obj" (Color 255 0 0)
  let bunny2 = Obj { vertices = mkArray (map (vmap (* 50.0)) (elems (vertices bunny))), faces = faces bunny, colors = colors bunny }
  let objects = [bunny2]

  let (image, _) = render width height objects origin direction fov
  save_image image "output/hs.ppm"
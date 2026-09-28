import struct
import os
import sys
import re

def parse_materials_and_textures(data):
    """
    Извлекает из tzMF имена материалов (*.m.*) 
    и сопоставляет их с наборами PBR-текстур (Albedo, Normal, Parameter/PBR).
    """
    # Ищем материалы
    mat_pattern = rb'([a-zA-Z0-9_\-]+\.m\.[a-zA-Z0-9_\-]+)'
    found_mats = [m.decode('utf-8', errors='ignore') for m in re.findall(mat_pattern, data, re.IGNORECASE)]
    
    unique_mats = []
    for m in found_mats:
        if m not in unique_mats:
            unique_mats.append(m)

    # Ищем все текстурные пути
    tex_pattern = rb'([a-zA-Z0-9_\-\/]+\.(?:png|tga|bmp|dds|texture))'
    found_texs = [t.decode('utf-8', errors='ignore') for t in re.findall(tex_pattern, data, re.IGNORECASE)]
    
    unique_texs = []
    for t in found_texs:
        clean_t = re.sub(r'\.texture$', '.png', t, flags=re.IGNORECASE)
        if clean_t not in unique_texs:
            unique_texs.append(clean_t)

    # Словарь сопоставления текстур для каждого материала
    mat_textures = {m: {'albedo': None, 'normal': None, 'param': None} for m in unique_mats}

    for m in unique_mats:
        base_prefix = m.split('.m.')[0]  # Название до .m. (например, metelnik, cabine, bort1)
        
        for tex in unique_texs:
            tex_lower = tex.lower()
            if base_prefix.lower() in tex_lower:
                if 'normal' in tex_lower or 'norm' in tex_lower or 'n.png' in tex_lower:
                    mat_textures[m]['normal'] = tex
                elif 'param' in tex_lower or 'pbr' in tex_lower or 'rough' in tex_lower or 'metal' in tex_lower:
                    mat_textures[m]['param'] = tex
                elif 'albedo' in tex_lower or 'diffuse' in tex_lower or 'color' in tex_lower or 'a.png' in tex_lower:
                    mat_textures[m]['albedo'] = tex

    return unique_mats, mat_textures, unique_texs

def parse_tzmf(file_path):
    if not os.path.exists(file_path):
        print(f"[-] Ошибка: Файл '{file_path}' не найден!")
        return False

    with open(file_path, 'rb') as f:
        data = f.read()

    if not data.startswith(b'tzMF'):
        print(f"[-] Ошибка: Файл '{file_path}' не содержит сигнатуру tzMF!")
        return False

    base_name = os.path.splitext(file_path)[0]
    obj_name = base_name + ".obj"
    mtl_name = base_name + ".mtl"

    stride_be = b'\x00\x00\x00\x2C'
    stride_le = b'\x2C\x00\x00\x00'

    all_vertices = []
    all_uvs = []
    all_normals = []
    mesh_groups = []

    vertex_index_offset = 0
    search_pos = 0x40
    chunk_index = 0

    while True:
        pos_be = data.find(stride_be, search_pos)
        pos_le = data.find(stride_le, search_pos)

        if pos_be == -1 and pos_le == -1:
            break
        elif pos_be != -1 and pos_le != -1:
            if pos_be < pos_le:
                stride_pos, is_big_endian = pos_be, True
            else:
                stride_pos, is_big_endian = pos_le, False
        elif pos_be != -1:
            stride_pos, is_big_endian = pos_be, True
        else:
            stride_pos, is_big_endian = pos_le, False

        header_offset = stride_pos - 8
        fmt = '>III' if is_big_endian else '<III'

        num_vertices, num_indices, vertex_stride = struct.unpack_from(fmt, data, header_offset)

        if vertex_stride == 44 and 0 < num_vertices < 1000000 and 0 < num_indices < 3000000:
            chunk_index += 1
            verts_start = header_offset + 12

            chunk_vertices = []
            chunk_uvs = []
            chunk_normals = []
            chunk_faces = []

            for i in range(num_vertices):
                offset = verts_start + i * vertex_stride
                if offset + vertex_stride > len(data):
                    break
                
                vdata = struct.unpack_from('<11f', data, offset)
                
                px, py, pz = vdata[0], vdata[1], vdata[2]
                nx, ny, nz = vdata[3], vdata[4], vdata[5]
                u, v       = vdata[9], vdata[10]

                chunk_vertices.append((-px, pz, py))
                chunk_normals.append((-nx, nz, ny))
                chunk_uvs.append((u, 1.0 - v))

            indices_start = verts_start + (num_vertices * vertex_stride)
            
            for i in range(0, num_indices, 3):
                idx_offset = indices_start + i * 2
                if idx_offset + 6 > len(data):
                    break
                i1, i2, i3 = struct.unpack_from('<HHH', data, idx_offset)
                
                chunk_faces.append((
                    i1 + 1 + vertex_index_offset,
                    i2 + 1 + vertex_index_offset,
                    i3 + 1 + vertex_index_offset
                ))

            all_vertices.extend(chunk_vertices)
            all_normals.extend(chunk_normals)
            all_uvs.extend(chunk_uvs)

            mesh_groups.append((f"SubMesh_{chunk_index}", chunk_faces))
            
            vertex_index_offset += len(chunk_vertices)
            search_pos = indices_start + (num_indices * 2)
        else:
            search_pos = stride_pos + 4

    if not mesh_groups:
        print(f"[-] Не удалось извлечь геометрию из {file_path}")
        return False

    extracted_mats, mat_textures, all_texs = parse_materials_and_textures(data)
    
    final_materials = []
    for i in range(len(mesh_groups)):
        if i < len(extracted_mats):
            mat_name = extracted_mats[i]
        else:
            mat_name = f"Material_{i + 1}"
        final_materials.append(mat_name)

    # Генерация расширенного MTL (3 текстуры на материал)
    with open(mtl_name, 'w', encoding='utf-8') as mtl_out:
        mtl_out.write(f"# Material Library for Trainz PBR Shader: {file_path}\n")
        
        for idx, mat_name in enumerate(final_materials):
            mtl_out.write(f"\nnewmtl {mat_name}\n")
            mtl_out.write("Ka 1.000 1.000 1.000\n")
            mtl_out.write("Kd 1.000 1.000 1.000\n")
            mtl_out.write("Ks 0.050 0.050 0.050\n")
            
            tex_info = mat_textures.get(mat_name, {})
            
            # 1. Albedo
            albedo = tex_info.get('albedo')
            if not albedo and idx < len(all_texs):
                albedo = all_texs[idx]
            if albedo:
                mtl_out.write(f"map_Kd {albedo}\n")
            
            # 2. Normal Map
            normal = tex_info.get('normal')
            if normal:
                mtl_out.write(f"map_Bump -bm 1.0 {normal}\n")
                mtl_out.write(f"bump {normal}\n")
                
            # 3. Parameters / Metallic-Roughness
            param = tex_info.get('param')
            if param:
                mtl_out.write(f"map_Pr {param}\n")  # Roughness Map
                mtl_out.write(f"map_Pm {param}\n")  # Metallic Map

    # Сохранение OBJ
    with open(obj_name, 'w', encoding='utf-8') as out:
        out.write(f"# Converted from TrainzMesh: {file_path}\n")
        out.write(f"mtllib {os.path.basename(mtl_name)}\n")
        
        for v in all_vertices:
            out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        for vt in all_uvs:
            out.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")
            
        for vn in all_normals:
            out.write(f"vn {vn[0]:.6f} {vn[1]:.6f}\n")

        for (group_name, faces), mat_name in zip(mesh_groups, final_materials):
            out.write(f"\ng {group_name}\n")
            out.write(f"usemtl {mat_name}\n")
            for f1, f2, f3 in faces:
                out.write(f"f {f1}/{f1}/{f1} {f2}/{f2}/{f2} {f3}/{f3}/{f3}\n")

    print(f"\n[+] {file_path}:")
    print(f"    Сабмешей: {len(mesh_groups)} | Вершин: {len(all_vertices)}")
    if extracted_mats:
        print(f"    Извлечено материалов: {len(extracted_mats)}")
    print(f"[=>] Экспортировано: {obj_name} и {mtl_name}")
    return True

def main():
    if len(sys.argv) < 2:
        print("\nИспользование:")
        print("  python con.py <имя_файла.trainzmesh>")
        print("  python con.py all\n")
        return

    arg = sys.argv[1].strip()

    if arg.lower() == 'all':
        tz_files = [f for f in os.listdir('.') if f.lower().endswith('.trainzmesh')]
        for f in sorted(tz_files):
            parse_tzmf(f)
    else:
        target_file = arg if arg.lower().endswith('.trainzmesh') else arg + '.trainzmesh'
        parse_tzmf(target_file)

if __name__ == '__main__':
    main()
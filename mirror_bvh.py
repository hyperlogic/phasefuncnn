import bvhio

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: expected bvh file argument")
        exit(1)

    mocap_filename = sys.argv[1]
    mocap_basename = os.path.splitext(os.path.basename(mocap_filename))[0]
    outbasepath = os.path.join(OUTPUT_DIR, mocap_basename)

    print(f"Loading {mocap_filename}")
    bvhio.readAsBvh(mocap_filename)


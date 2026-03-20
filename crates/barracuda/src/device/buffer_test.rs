// SPDX-License-Identifier: AGPL-3.0-or-later
//! Direct buffer test - bypass Tensor abstraction

#[tokio::test]
async fn test_direct_buffer_write_read() {
    use wgpu::util::DeviceExt;
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else { return };
    
    println!("\n=== Testing Direct Buffer Write/Read ===\n");
    
    // Create data
    let data = vec![1.0f32; 27];
    println!("Input: {} elements, all 1.0", data.len());
    
    // Create buffer directly
    let buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Buffer"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    
    println!("Buffer created: {} bytes", data.len() * 4);
    
    // Read back immediately
    let result = device.read_buffer_f32(&buffer, 27).unwrap();
    println!("Read back: {} elements", result.len());
    println!("First 10: {:?}", &result[0..10]);
    
    // Validate
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, 1.0, "Index {i}: expected 1.0, got {val}");
    }
    
    println!("✅ Direct buffer test passed!");
}

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.roboflow_dataset import RoboflowFarmlandDataset

def test_dataset_loading():
    """测试数据集加载"""
    print("🧪 开始测试数据集加载...")
    
    dataset_yaml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'dataset.yaml'
    )
    
    print(f"📁 数据集配置文件: {dataset_yaml_path}")
    print(f"📁 文件存在: {os.path.exists(dataset_yaml_path)}")
    
    try:
        train_dataset = RoboflowFarmlandDataset(
            data_yaml_path=dataset_yaml_path,
            split='train',
            image_size=(512, 512),
            augment=False,
            cache_images=False
        )
        
        print(f"\n✅ 训练集加载成功!")
        print(f"📊 数据集大小: {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"🖼️  样本图像形状: {sample['image'].shape}")
            print(f"🏷️  样本标签形状: {sample['labels'].shape}")
            print(f"📸 样本路径: {sample['image_path']}")
            print(f"🎯 目标数量: {sample['num_objects']}")
        
        val_dataset = RoboflowFarmlandDataset(
            data_yaml_path=dataset_yaml_path,
            split='val',
            image_size=(512, 512),
            augment=False,
            cache_images=False
        )
        
        print(f"\n✅ 验证集加载成功!")
        print(f"📊 数据集大小: {len(val_dataset)}")
        
        print("\n🎉 数据集测试完成!")
        return True
        
    except Exception as e:
        print(f"\n❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
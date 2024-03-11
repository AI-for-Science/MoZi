# 墨子（MoZi）: An IP-oriented Multilingual Large Language Model
![image](https://github.com/AI-for-Science/MoZi/assets/56249874/0d0f8faa-4074-4748-b395-481517aea34e)

## Paper
[MoZIP: A Multilingual Benchmark to Evaluate Large Language Models in Intellectual Property](https://arxiv.org/abs/2402.16389) (Accepted by LREC-COLING 2024)
```
@article{ni2024mozip,
  title={MoZIP: A Multilingual Benchmark to Evaluate Large Language Models in Intellectual Property},
  author={Ni, Shiwen and Tan, Minghuan and Bai, Yuelin and Niu, Fuqiang and Yang, Min and Zhang, Bowen and Xu, Ruifeng and Chen, Xiaojun and Li, Chengming and Hu, Xiping and others},
  journal={arXiv preprint arXiv:2402.16389},
  year={2024}
}
```

## Model
（基于chatglm3-6b微调的mozi2-6b）https://huggingface.co/Nozomi7/mozi2-6b

（基于baichuan2-7b微调的mozi2-7b）https://huggingface.co/Nozomi7/mozi2-7b

（基于bloomz-mt-7b微调的mozi-7b）https://huggingface.co/BNNT/mozi-7b

## MoZIP benchmark
https://huggingface.co/datasets/BNNT/IPQuiz

https://huggingface.co/datasets/BNNT/IPQA

https://huggingface.co/datasets/BNNT/PatentMatch

## STF data
General STF: https://huggingface.co/datasets/BNNT/mozi_general_instructions_3m

Ip-specific STF: https://huggingface.co/datasets/BNNT/mozi_IP_instructions

## Example
#### Q：你知道怎么引用他人作品又不侵犯著作权吗？

**chatglm3-6b**：引用他人作品时，我们需要遵循合理使用原则，遵循以下几点，可以在一定程度上避免侵犯著作权：  
1. 获取授权：在使用他人作品之前，尽量尝试与原作者取得联系，获得授权。email或者社交媒体等方式联系作者，征得他们的同意。  
2. 注明来源：在使用他人作品时，应在使用过程中注明作者姓名、作品名称以及来源。这不仅是尊重原作者的劳动成果，也是遵守著作权法的表现。  
3. transform 或者改编：在引用他人作品时，可以进行一定的修改或者改编，让作品更适合自己的需求。这种方式可以降低侵权风险。  
4. 使用引用：如果引用他人作品是为了学术研究、评论、新闻报道等目的，那么可以不用获得原作者的授权。  
总结起来，要想合理使用他人作品，就需要遵循上述原则，尽可能地避免侵犯著作权。  

**mozi2-6b**：引用他人作品是一种常见的学术或商业行为，但在引用过程中需要遵守一些法律和道德准则，以避免侵犯著作权。以下是一些建议：  
1. 尊重版权：在引用他人作品时，应始终尊重版权持有人的权利和知识产权。未经许可，不要擅自使用他人的作品，以免侵犯他人的著作权。  
2. 注明来源：在引用他人作品时，应确保注明来源，并使用引号括起所引用的部分。这有助于表明你对版权的尊重，并减少侵权的风险。  
3. 合理使用：在某些情况下，对他人作品的使用可能是合理的，例如为了评论、批评、新闻报道或学术研究等目的。但是，这种使用必须在合理范围内，并不得影响版权持有人的利益。  
4. 付费使用：如果你需要使用他人的作品，而该作品已经发表，但未获得版权持有人的许可或授权，那么你可能需要考虑付费使用。在这种情况下，你应该与版权持有人联系并协商许可条款。  
5. 避免侵权：在引用他人作品时，要避免使用他人的作品作为自己的创作成果，以避免侵权纠纷。如果你需要在自己的创作中使用他人的作品，应该确保你的使用方式符合版权法律法规的规定。  
总之，在引用他人作品时，我们应该尊重版权，注明来源，合理使用，付费使用，并避免侵权。这样可以保护自己的合法权益，同时也有助于促进知识产权的保护和市场的发展。  

#### Q：商号与商标的区别是什么？

**chatglm3-6b**：商号和商标都是用于区分和识别某个企业或产品的标志，但它们之间有一些区别。  
商号是指企业为了区分自己的商品或服务而使用的名称，通常是文字、字母、数字或组合等形式。商号不一定是独特的，可能会有多个企业使用相同的商号。商号主要起到标识作用，让消费者能够识别商品或服务的来源。  
商标是指通过使用特定的标志，如文字、图像、符号、设计等来区分和识别某个企业或产品的标志。商标通常是独特的，只能由一个企业使用，有助于消费者在市场上区分和选择产品或服务。商标还可以包括企业的名称、标志、口号等。  
商号和商标之间的主要区别在于是否具有独特性。商号可能会有多个企业使用相同或相似的名称，而商标必须是独特的，只能由一个企业使用。此外，商号主要起到标识作用，而商标除了标识作用外，还具有区分和选择产品的功能。

**mozi2-6b**：
商号与商标在以下六个方面存在区别：
1. 功能和作用：商号主要用于区分不同的企业，而商标则主要用于区分不同的商品。一般而言，一个公司和企业只能有一个商号，而可以根据不同的商品来注册商标。
2. 表现形式：商号一般只能用两个以上的汉字组成，而商标则可以采用数字、图形、拼音、颜色和英文字母等来组合运用。这也反映了两者在表现形式上的差异。
3. 使用的范围和效力：在中国，商标权具有全国范围内的效力，且其有效期为十年，过期后可以进行续展。相对地，商号权具有一定的区域限制，首先要通过国家工商行政管理总局核准后在一定的区域内使用，但对于商号的使用期限这没有限制，即只要一经通过就可终身使用。
4. 转让和许可：商号权一般不能单独转让，而商标权可以单独转让或许可他人使用。此外，商标权的使用期限一般为十年，到期后可以续展，而商号权则没有这一限制。
5. 权利内容：商号权的内容一般包括禁止他人使用相同或近似商号，而商标权的内容则包括禁止他人使用相同或近似商标。此外，商标权还包括禁止他人使用与注册商标相同或近似的标志。
6. 法律保护：商号和商标都受法律保护，但受保护的范围和所依据的法律可能有所不同。在中国，商标受《中华人民共和国商标法》保护，而商号受《中华人民共和国公司法》和《中华人民共和国企业名称登记管理规定》保护。  
总的来说，商号和商标在功能、表现形式、使用范围、转让和许可、权利内容和法律保护等方面存在明显的区别。这些区别反映了两者在商业实践和法律规定上的差异。



